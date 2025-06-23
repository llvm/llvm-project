//===--- UnhandledExceptionAtNewCheck.cpp - clang-tidy --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnhandledExceptionAtNewCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {
namespace {

AST_MATCHER_P(CXXTryStmt, hasHandlerFor,
              ast_matchers::internal::Matcher<QualType>, InnerMatcher) {
  for (unsigned NH = Node.getNumHandlers(), I = 0; I < NH; ++I) {
    const CXXCatchStmt *CatchS = Node.getHandler(I);
    // Check for generic catch handler (match anything).
    if (CatchS->getCaughtType().isNull())
      return true;
    ast_matchers::internal::BoundNodesTreeBuilder Result(*Builder);
    if (InnerMatcher.matches(CatchS->getCaughtType(), Finder, &Result)) {
      *Builder = std::move(Result);
      return true;
    }
  }
  return false;
}

AST_MATCHER(CXXNewExpr, mayThrow) {
  FunctionDecl *OperatorNew = Node.getOperatorNew();
  if (!OperatorNew)
    return false;
  return !OperatorNew->getType()->castAs<FunctionProtoType>()->isNothrow();
}

} // namespace

UnhandledExceptionAtNewCheck::UnhandledExceptionAtNewCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void UnhandledExceptionAtNewCheck::registerMatchers(MatchFinder *Finder) {
  auto BadAllocType =
      recordType(hasDeclaration(cxxRecordDecl(hasName("::std::bad_alloc"))));
  auto ExceptionType =
      recordType(hasDeclaration(cxxRecordDecl(hasName("::std::exception"))));
  auto BadAllocReferenceType = referenceType(pointee(BadAllocType));
  auto ExceptionReferenceType = referenceType(pointee(ExceptionType));

  auto CatchBadAllocType =
      qualType(hasCanonicalType(anyOf(BadAllocType, BadAllocReferenceType,
                                      ExceptionType, ExceptionReferenceType)));
  auto BadAllocCatchingTryBlock = cxxTryStmt(hasHandlerFor(CatchBadAllocType));

  auto FunctionMayNotThrow = functionDecl(isNoThrow());

  Finder->addMatcher(cxxNewExpr(mayThrow(),
                                unless(hasAncestor(BadAllocCatchingTryBlock)),
                                hasAncestor(FunctionMayNotThrow))
                         .bind("new-expr"),
                     this);
}

void UnhandledExceptionAtNewCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedExpr = Result.Nodes.getNodeAs<CXXNewExpr>("new-expr");
  if (MatchedExpr)
    diag(MatchedExpr->getBeginLoc(),
         "missing exception handler for allocation failure at 'new'");
}

} // namespace clang::tidy::bugprone
