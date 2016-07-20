//===--- UseEmplaceCheck.cpp - clang-tidy----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UseEmplaceCheck.h"
#include "../utils/Matchers.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

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
      on(hasType(cxxRecordDecl(hasAnyName("std::vector", "llvm::SmallVector",
                                          "std::list", "std::deque")))));

  // We can't replace push_backs of smart pointer because
  // if emplacement fails (f.e. bad_alloc in vector) we will have leak of
  // passed pointer because smart pointer won't be constructed
  // (and destructed) as in push_back case.
  auto isCtorOfSmartPtr = hasDeclaration(cxxConstructorDecl(
      ofClass(hasAnyName("std::shared_ptr", "std::unique_ptr", "std::auto_ptr",
                         "std::weak_ptr"))));

  // Bitfields binds only to consts and emplace_back take it by universal ref.
  auto bitFieldAsArgument = hasAnyArgument(ignoringParenImpCasts(
      memberExpr(hasDeclaration(fieldDecl(matchers::isBitfield())))));

  // We could have leak of resource.
  auto newExprAsArgument = hasAnyArgument(ignoringParenImpCasts(cxxNewExpr()));
  auto constructingDerived =
      hasParent(implicitCastExpr(hasCastKind(CastKind::CK_DerivedToBase)));

  auto hasInitList = has(ignoringParenImpCasts(initListExpr()));
  auto soughtConstructExpr =
      cxxConstructExpr(
          unless(anyOf(isCtorOfSmartPtr, hasInitList, bitFieldAsArgument,
                       newExprAsArgument, constructingDerived,
                       has(materializeTemporaryExpr(hasInitList)))))
          .bind("ctor");
  auto hasConstructExpr = has(ignoringParenImpCasts(soughtConstructExpr));

  auto ctorAsArgument = materializeTemporaryExpr(
      anyOf(hasConstructExpr, has(cxxFunctionalCastExpr(hasConstructExpr))));

  Finder->addMatcher(
      cxxMemberCallExpr(callPushBack, has(ctorAsArgument)).bind("call"), this);
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
  auto CtorCallSourceRange = CharSourceRange::getCharRange(
      InnerCtorCall->getExprLoc(),
      CallParensRange.getBegin().getLocWithOffset(1));

  Diag << FixItHint::CreateRemoval(CtorCallSourceRange)
       << FixItHint::CreateRemoval(CharSourceRange::getCharRange(
              CallParensRange.getEnd(),
              CallParensRange.getEnd().getLocWithOffset(1)));
}

} // namespace modernize
} // namespace tidy
} // namespace clang
