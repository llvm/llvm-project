//===--- Matchers.h - clang-tidy-------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_MATCHERS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_MATCHERS_H

#include "clang/ASTMatchers/ASTMatchers.h"
#include "TypeTraits.h"

namespace clang {
namespace tidy {
namespace matchers {

AST_MATCHER(BinaryOperator, isRelationalOperator) {
  return Node.isRelationalOp();
}

AST_MATCHER(BinaryOperator, isEqualityOperator) {
  return Node.isEqualityOp();
}

AST_MATCHER(BinaryOperator, isComparisonOperator) {
  return Node.isComparisonOp();
}

AST_MATCHER(QualType, isExpensiveToCopy) {
  llvm::Optional<bool> IsExpensive =
      utils::type_traits::isExpensiveToCopy(Node, Finder->getASTContext());
  return IsExpensive && *IsExpensive;
}

AST_MATCHER(RecordDecl, isTriviallyDefaultConstructible) {
  return utils::type_traits::recordIsTriviallyDefaultConstructible(
      Node, Finder->getASTContext());
}

// Returns QualType matcher for references to const.
AST_MATCHER_FUNCTION(ast_matchers::TypeMatcher, isReferenceToConst) {
  using namespace ast_matchers;
  return referenceType(pointee(qualType(isConstQualified())));
}

} // namespace matchers
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_MATCHERS_H
