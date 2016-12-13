//===--- RedundantFunctionPtrDereferenceCheck.cpp - clang-tidy-------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RedundantFunctionPtrDereferenceCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

void RedundantFunctionPtrDereferenceCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(unaryOperator(hasOperatorName("*"),
                                   has(implicitCastExpr(
                                       hasCastKind(CK_FunctionToPointerDecay))))
                         .bind("op"),
                     this);
}

void RedundantFunctionPtrDereferenceCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Operator = Result.Nodes.getNodeAs<UnaryOperator>("op");
  diag(Operator->getOperatorLoc(),
       "redundant repeated dereference of function pointer")
      << FixItHint::CreateRemoval(Operator->getOperatorLoc());
}

} // namespace readability
} // namespace tidy
} // namespace clang
