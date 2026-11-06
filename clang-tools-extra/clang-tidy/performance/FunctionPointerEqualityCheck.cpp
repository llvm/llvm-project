//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FunctionPointerEqualityCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <iostream>

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

void FunctionPointerEqualityCheck::registerMatchers(MatchFinder *Finder) {
  // Matching ptr == ptr, but not ptr == nullptr, or ptr == integer (function
  // pointers to 1 and -1 are frequently used for "invalid" function pointers).
  // Also match equality comparisons of member pointers that point to
  // functions.
  const auto FunctionPointerType = hasCanonicalType(anyOf(
      pointsTo(functionType()), memberPointerType(pointee(functionType()))));
  Finder->addMatcher(
      binaryOperator(hasAnyOperatorName("==", "!="),
                     hasLHS(hasType(FunctionPointerType)),
                     hasRHS(hasType(FunctionPointerType)),
                     unless(hasEitherOperand(ignoringParenCasts(
                         anyOf(nullPointerConstant(), hasType(isInteger()))))),
                     unless(isInTemplateInstantiation()))
          .bind("op"),
      this);

  // Note, that while some of these containers check equality through the
  // `Compare` requirement, the `Compare` requirement mandates strict weak
  // ordering, which is unspecified behaviour anyways. So even though they
  // aren't an issue with --icf=all, they're still discouraged.
  Finder->addMatcher(
      declaratorDecl(
          hasType(hasCanonicalType(
              recordType(hasDeclaration(classTemplateSpecializationDecl(
                  hasAnyName(
                      "::std::flat_map", "::std::flat_set", "::std::map",
                      "::std::multimap", "::std::multiset", "::std::set",
                      "::std::flat_multimap", "::std::flat_multiset",
                      "::std::unordered_map", "::std::unordered_multimap",
                      "::std::unordered_multiset", "::std::unordered_set"),
                  hasTemplateArgument(0,
                                      refersToType(FunctionPointerType))))))),
          unless(anyOf(isInstantiated(), hasAncestor(decl(isInstantiated())))))
          .bind("var"),
      this);
}

void FunctionPointerEqualityCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (const auto *MatchedOp = Result.Nodes.getNodeAs<BinaryOperator>("op")) {
    diag(MatchedOp->getOperatorLoc(),
         "function pointer equality comparison is unsound with '--icf=all' "
         "enabled");
    return;
  }
  if (const auto *MatchedDecl = Result.Nodes.getNodeAs<DeclaratorDecl>("var")) {
    if (MatchedDecl->getDeclName()) {
      diag(MatchedDecl->getLocation(), "associative or unordered container %0 "
                                       "with a function pointer key type "
                                       "is unsound with '--icf=all' enabled")
          << MatchedDecl;
    } else {
      diag(MatchedDecl->getLocation(), "associative or unordered container "
                                       "with a function pointer key type is "
                                       "unsound with '--icf=all' enabled");
    }
  }
}

} // namespace clang::tidy::performance
