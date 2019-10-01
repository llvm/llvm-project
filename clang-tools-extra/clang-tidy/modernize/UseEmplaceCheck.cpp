//===--- UseEmplaceCheck.cpp - clang-tidy----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseEmplaceCheck.h"
#include "../utils/OptionsUtils.h"
using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

namespace {
AST_MATCHER(DeclRefExpr, hasExplicitTemplateArgs) {
  return Node.hasExplicitTemplateArgs();
}

const auto DefaultContainersWithPushBack =
    "::std::vector; ::std::list; ::std::deque";
const auto DefaultSmartPointers =
    "::std::shared_ptr; ::std::unique_ptr; ::std::auto_ptr; ::std::weak_ptr";
const auto DefaultTupleTypes = "::std::pair; ::std::tuple";
const auto DefaultTupleMakeFunctions = "::std::make_pair; ::std::make_tuple";
} // namespace

UseEmplaceCheck::UseEmplaceCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreImplicitConstructors(Options.get("IgnoreImplicitConstructors", 0)),
      ContainersWithPushBack(utils::options::parseStringList(Options.get(
          "ContainersWithPushBack", DefaultContainersWithPushBack))),
      SmartPointers(utils::options::parseStringList(
          Options.get("SmartPointers", DefaultSmartPointers))),
      TupleTypes(utils::options::parseStringList(
          Options.get("TupleTypes", DefaultTupleTypes))),
      TupleMakeFunctions(utils::options::parseStringList(
          Options.get("TupleMakeFunctions", DefaultTupleMakeFunctions))) {}

void UseEmplaceCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus11)
    return;

  // FIXME: Bunch of functionality that could be easily added:
  // + add handling of `push_front` for std::forward_list, std::list
  // and std::deque.
  // + add handling of `push` for std::stack, std::queue, std::priority_queue
  // + add handling of `insert` for stl associative container, but be careful
  // because this requires special treatment (it could cause performance
  // regression)
  // + match for emplace calls that should be replaced with insertion
  auto CallPushBack = cxxMemberCallExpr(
      hasDeclaration(functionDecl(hasName("push_back"))),
      on(hasType(cxxRecordDecl(hasAnyName(SmallVector<StringRef, 5>(
          ContainersWithPushBack.begin(), ContainersWithPushBack.end()))))));

  // We can't replace push_backs of smart pointer because
  // if emplacement fails (f.e. bad_alloc in vector) we will have leak of
  // passed pointer because smart pointer won't be constructed
  // (and destructed) as in push_back case.
  auto IsCtorOfSmartPtr = hasDeclaration(cxxConstructorDecl(ofClass(hasAnyName(
      SmallVector<StringRef, 5>(SmartPointers.begin(), SmartPointers.end())))));

  // Bitfields binds only to consts and emplace_back take it by universal ref.
  auto BitFieldAsArgument = hasAnyArgument(
      ignoringImplicit(memberExpr(hasDeclaration(fieldDecl(isBitField())))));

  // Initializer list can't be passed to universal reference.
  auto InitializerListAsArgument = hasAnyArgument(
      ignoringImplicit(cxxConstructExpr(isListInitialization())));

  // We could have leak of resource.
  auto NewExprAsArgument = hasAnyArgument(ignoringImplicit(cxxNewExpr()));
  // We would call another constructor.
  auto ConstructingDerived =
      hasParent(implicitCastExpr(hasCastKind(CastKind::CK_DerivedToBase)));

  // emplace_back can't access private constructor.
  auto IsPrivateCtor = hasDeclaration(cxxConstructorDecl(isPrivate()));

  auto HasInitList = anyOf(has(ignoringImplicit(initListExpr())),
                           has(cxxStdInitializerListExpr()));

  // FIXME: Discard 0/NULL (as nullptr), static inline const data members,
  // overloaded functions and template names.
  auto SoughtConstructExpr =
      cxxConstructExpr(
          unless(anyOf(IsCtorOfSmartPtr, HasInitList, BitFieldAsArgument,
                       InitializerListAsArgument, NewExprAsArgument,
                       ConstructingDerived, IsPrivateCtor)))
          .bind("ctor");
  auto HasConstructExpr = has(ignoringImplicit(SoughtConstructExpr));

  auto MakeTuple = ignoringImplicit(
      callExpr(
          callee(expr(ignoringImplicit(declRefExpr(
              unless(hasExplicitTemplateArgs()),
              to(functionDecl(hasAnyName(SmallVector<StringRef, 2>(
                  TupleMakeFunctions.begin(), TupleMakeFunctions.end())))))))))
          .bind("make"));

  // make_something can return type convertible to container's element type.
  // Allow the conversion only on containers of pairs.
  auto MakeTupleCtor = ignoringImplicit(cxxConstructExpr(
      has(materializeTemporaryExpr(MakeTuple)),
      hasDeclaration(cxxConstructorDecl(ofClass(hasAnyName(
          SmallVector<StringRef, 2>(TupleTypes.begin(), TupleTypes.end())))))));

  auto SoughtParam = materializeTemporaryExpr(
      anyOf(has(MakeTuple), has(MakeTupleCtor),
            HasConstructExpr, has(cxxFunctionalCastExpr(HasConstructExpr))));

  Finder->addMatcher(cxxMemberCallExpr(CallPushBack, has(SoughtParam),
                                       unless(isInTemplateInstantiation()))
                         .bind("call"),
                     this);
}

void UseEmplaceCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CXXMemberCallExpr>("call");
  const auto *CtorCall = Result.Nodes.getNodeAs<CXXConstructExpr>("ctor");
  const auto *MakeCall = Result.Nodes.getNodeAs<CallExpr>("make");
  assert((CtorCall || MakeCall) && "No push_back parameter matched");

  if (IgnoreImplicitConstructors && CtorCall && CtorCall->getNumArgs() >= 1 &&
      CtorCall->getArg(0)->getSourceRange() == CtorCall->getSourceRange())
    return;

  const auto FunctionNameSourceRange = CharSourceRange::getCharRange(
      Call->getExprLoc(), Call->getArg(0)->getExprLoc());

  auto Diag = diag(Call->getExprLoc(), "use emplace_back instead of push_back");

  if (FunctionNameSourceRange.getBegin().isMacroID())
    return;

  const auto *EmplacePrefix = MakeCall ? "emplace_back" : "emplace_back(";
  Diag << FixItHint::CreateReplacement(FunctionNameSourceRange, EmplacePrefix);

  const SourceRange CallParensRange =
      MakeCall ? SourceRange(MakeCall->getCallee()->getEndLoc(),
                             MakeCall->getRParenLoc())
               : CtorCall->getParenOrBraceRange();

  // Finish if there is no explicit constructor call.
  if (CallParensRange.getBegin().isInvalid())
    return;

  const SourceLocation ExprBegin =
      MakeCall ? MakeCall->getExprLoc() : CtorCall->getExprLoc();

  // Range for constructor name and opening brace.
  const auto ParamCallSourceRange =
      CharSourceRange::getTokenRange(ExprBegin, CallParensRange.getBegin());

  Diag << FixItHint::CreateRemoval(ParamCallSourceRange)
       << FixItHint::CreateRemoval(CharSourceRange::getTokenRange(
           CallParensRange.getEnd(), CallParensRange.getEnd()));
}

void UseEmplaceCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "ContainersWithPushBack",
                utils::options::serializeStringList(ContainersWithPushBack));
  Options.store(Opts, "SmartPointers",
                utils::options::serializeStringList(SmartPointers));
  Options.store(Opts, "TupleTypes",
                utils::options::serializeStringList(TupleTypes));
  Options.store(Opts, "TupleMakeFunctions",
                utils::options::serializeStringList(TupleMakeFunctions));
}

} // namespace modernize
} // namespace tidy
} // namespace clang
