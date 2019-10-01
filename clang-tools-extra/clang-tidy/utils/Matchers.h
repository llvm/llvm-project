//===--- Matchers.h - clang-tidy-------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_MATCHERS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_MATCHERS_H

#include "TypeTraits.h"
#include "clang/ASTMatchers/ASTMatchers.h"

namespace clang {
namespace tidy {
namespace matchers {

AST_MATCHER(BinaryOperator, isAssignmentOperator) {
  return Node.isAssignmentOp();
}

AST_MATCHER(BinaryOperator, isRelationalOperator) {
  return Node.isRelationalOp();
}

AST_MATCHER(BinaryOperator, isEqualityOperator) { return Node.isEqualityOp(); }

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

AST_MATCHER_P(NamedDecl, matchesAnyListedName, std::vector<std::string>,
              NameList) {
  return llvm::any_of(NameList, [&Node](const std::string &Name) {
      return llvm::Regex(Name).match(Node.getName());
    });
}

} // namespace matchers
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_MATCHERS_H
