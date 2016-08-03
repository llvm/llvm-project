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

static const auto DefaultContainersWithPushBack =
    "::std::vector; ::std::list; ::std::deque";
static const auto DefaultSmartPointers =
    "::std::shared_ptr; ::std::unique_ptr; ::std::auto_ptr; ::std::weak_ptr";

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
  // + match for make_pair calls.
  auto callPushBack = cxxMemberCallExpr(
      hasDeclaration(functionDecl(hasName("push_back"))),
      on(hasType(cxxRecordDecl(hasAnyName(SmallVector<StringRef, 5>(
          ContainersWithPushBack.begin(), ContainersWithPushBack.end()))))));

  // We can't replace push_backs of smart pointer because
  // if emplacement fails (f.e. bad_alloc in vector) we will have leak of
  // passed pointer because smart pointer won't be constructed
  // (and destructed) as in push_back case.
  auto isCtorOfSmartPtr = hasDeclaration(cxxConstructorDecl(ofClass(hasAnyName(
      SmallVector<StringRef, 5>(SmartPointers.begin(), SmartPointers.end())))));

  // Bitfields binds only to consts and emplace_back take it by universal ref.
  auto bitFieldAsArgument = hasAnyArgument(
      ignoringImplicit(memberExpr(hasDeclaration(fieldDecl(isBitField())))));

  // Initializer list can't be passed to universal reference.
  auto initializerListAsArgument = hasAnyArgument(
      ignoringImplicit(cxxConstructExpr(isListInitialization())));

  // We could have leak of resource.
  auto newExprAsArgument = hasAnyArgument(ignoringImplicit(cxxNewExpr()));
  // We would call another constructor.
  auto constructingDerived =
      hasParent(implicitCastExpr(hasCastKind(CastKind::CK_DerivedToBase)));

  // emplace_back can't access private constructor.
  auto isPrivateCtor = hasDeclaration(cxxConstructorDecl(isPrivate()));

  auto hasInitList = has(ignoringImplicit(initListExpr()));
  // FIXME: Discard 0/NULL (as nullptr), static inline const data members,
  // overloaded functions and template names.
  auto soughtConstructExpr =
      cxxConstructExpr(
          unless(anyOf(isCtorOfSmartPtr, hasInitList, bitFieldAsArgument,
                       initializerListAsArgument, newExprAsArgument,
                       constructingDerived, isPrivateCtor)))
          .bind("ctor");
  auto hasConstructExpr = has(ignoringImplicit(soughtConstructExpr));

  auto ctorAsArgument = materializeTemporaryExpr(
      anyOf(hasConstructExpr, has(cxxFunctionalCastExpr(hasConstructExpr))));

  Finder->addMatcher(cxxMemberCallExpr(callPushBack, has(ctorAsArgument),
                                       unless(isInTemplateInstantiation()))
                         .bind("call"),
                     this);
}

void UseEmplaceCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CXXMemberCallExpr>("call");
  const auto *InnerCtorCall = Result.Nodes.getNodeAs<CXXConstructExpr>("ctor");

  auto FunctionNameSourceRange = CharSourceRange::getCharRange(
      Call->getExprLoc(), Call->getArg(0)->getExprLoc());

  auto Diag = diag(Call->getExprLoc(), "use emplace_back instead of push_back");

  if (FunctionNameSourceRange.getBegin().isMacroID())
    return;

  Diag << FixItHint::CreateReplacement(FunctionNameSourceRange,
                                       "emplace_back(");

  auto CallParensRange = InnerCtorCall->getParenOrBraceRange();

  // Finish if there is no explicit constructor call.
  if (CallParensRange.getBegin().isInvalid())
    return;

  // Range for constructor name and opening brace.
  auto CtorCallSourceRange = CharSourceRange::getTokenRange(
      InnerCtorCall->getExprLoc(), CallParensRange.getBegin());

  Diag << FixItHint::CreateRemoval(CtorCallSourceRange)
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
