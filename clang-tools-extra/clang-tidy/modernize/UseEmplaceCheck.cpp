//===--- UseEmplaceCheck.cpp - clang-tidy----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
} // namespace

UseEmplaceCheck::UseEmplaceCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      ContainersWithPushBack(utils::options::parseStringList(Options.get(
          "ContainersWithPushBack", DefaultContainersWithPushBack))),
      SmartPointers(utils::options::parseStringList(
          Options.get("SmartPointers", DefaultSmartPointers))) {}

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

  auto HasInitList = has(ignoringImplicit(initListExpr()));
  // FIXME: Discard 0/NULL (as nullptr), static inline const data members,
  // overloaded functions and template names.
  auto SoughtConstructExpr =
      cxxConstructExpr(
          unless(anyOf(IsCtorOfSmartPtr, HasInitList, BitFieldAsArgument,
                       InitializerListAsArgument, NewExprAsArgument,
                       ConstructingDerived, IsPrivateCtor)))
          .bind("ctor");
  auto HasConstructExpr = has(ignoringImplicit(SoughtConstructExpr));

  auto MakePair = ignoringImplicit(
      callExpr(callee(expr(ignoringImplicit(
          declRefExpr(unless(hasExplicitTemplateArgs()),
                      to(functionDecl(hasName("::std::make_pair"))))
      )))).bind("make_pair"));

  // make_pair can return type convertible to container's element type.
  // Allow the conversion only on containers of pairs.
  auto MakePairCtor = ignoringImplicit(cxxConstructExpr(
      has(materializeTemporaryExpr(MakePair)),
      hasDeclaration(cxxConstructorDecl(ofClass(hasName("::std::pair"))))));

  auto SoughtParam = materializeTemporaryExpr(
      anyOf(has(MakePair), has(MakePairCtor),
            HasConstructExpr, has(cxxFunctionalCastExpr(HasConstructExpr))));

  Finder->addMatcher(cxxMemberCallExpr(CallPushBack, has(SoughtParam),
                                       unless(isInTemplateInstantiation()))
                         .bind("call"),
                     this);
}

void UseEmplaceCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CXXMemberCallExpr>("call");
  const auto *InnerCtorCall = Result.Nodes.getNodeAs<CXXConstructExpr>("ctor");
  const auto *MakePairCall = Result.Nodes.getNodeAs<CallExpr>("make_pair");
  assert((InnerCtorCall || MakePairCall) && "No push_back parameter matched");

  const auto FunctionNameSourceRange = CharSourceRange::getCharRange(
      Call->getExprLoc(), Call->getArg(0)->getExprLoc());

  auto Diag = diag(Call->getExprLoc(), "use emplace_back instead of push_back");

  if (FunctionNameSourceRange.getBegin().isMacroID())
    return;

  const auto *EmplacePrefix = MakePairCall ? "emplace_back" : "emplace_back(";
  Diag << FixItHint::CreateReplacement(FunctionNameSourceRange, EmplacePrefix);

  const SourceRange CallParensRange =
      MakePairCall ? SourceRange(MakePairCall->getCallee()->getLocEnd(),
                                 MakePairCall->getRParenLoc())
                   : InnerCtorCall->getParenOrBraceRange();

  // Finish if there is no explicit constructor call.
  if (CallParensRange.getBegin().isInvalid())
    return;

  const SourceLocation ExprBegin =
      MakePairCall ? MakePairCall->getExprLoc() : InnerCtorCall->getExprLoc();

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
}

} // namespace modernize
} // namespace tidy
} // namespace clang
