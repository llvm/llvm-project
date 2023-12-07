//===--- RedundantFunctionPtrDereferenceCheck.cpp - clang-tidy-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantFunctionPtrDereferenceCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

void RedundantFunctionPtrDereferenceCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      traverse(TK_AsIs, unaryOperator(hasOperatorName("*"),
                                      has(implicitCastExpr(hasCastKind(
                                          CK_FunctionToPointerDecay))))
                            .bind("op")),
      this);
}

void RedundantFunctionPtrDereferenceCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Operator = Result.Nodes.getNodeAs<UnaryOperator>("op");
  diag(Operator->getOperatorLoc(),
       "redundant repeated dereference of function pointer")
      << FixItHint::CreateRemoval(Operator->getOperatorLoc());
}

} // namespace clang::tidy::readability
