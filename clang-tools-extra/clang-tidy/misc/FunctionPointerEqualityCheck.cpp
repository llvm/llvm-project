//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FunctionPointerEqualityCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

FunctionPointerEqualityCheck::FunctionPointerEqualityCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      KeyedContainerTypes(utils::options::parseStringList(
          Options.get("KeyedContainerTypes", "::std::flat_map;"
                                             "::std::flat_set;"
                                             "::std::map;"
                                             "::std::multimap;"
                                             "::std::multiset;"
                                             "::std::set;"
                                             "::std::flat_multimap;"
                                             "::std::flat_multiset;"
                                             "::std::unordered_map;"
                                             "::std::unordered_multimap;"
                                             "::std::unordered_multiset;"
                                             "::std::unordered_set"))) {}

void FunctionPointerEqualityCheck::registerMatchers(MatchFinder *Finder) {
  const auto CompileTimeOrUnevaluated =
      anyOf(hasAncestor(expr(matchers::hasUnevaluatedContext())),
            hasAncestor(staticAssertDecl()));
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
                     unless(CompileTimeOrUnevaluated))
          .bind("op"),
      this);

  // Note, that while some of these containers check equality through the
  // `Compare` requirement, the `Compare` requirement mandates strict weak
  // ordering, which is unspecified behaviour or implementation defined
  // anyways when they don't default to std::less.
  const auto ContainerTypesMatcher =
      matchers::matchesAnyListedRegexName(KeyedContainerTypes);
  Finder->addMatcher(
      declaratorDecl(hasType(hasCanonicalType(recordType(
                         hasDeclaration(classTemplateSpecializationDecl(
                             ContainerTypesMatcher,
                             hasTemplateArgument(
                                 0, refersToType(FunctionPointerType))))))))
          .bind("var"),
      this);
}

void FunctionPointerEqualityCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (const auto *MatchedOp = Result.Nodes.getNodeAs<BinaryOperator>("op")) {
    // Skip if either type is of type decltype(nullptr). This is a builtin
    // type.
    if (MatchedOp->getLHS()->IgnoreParenCasts()->getType()->isNullPtrType() ||
        MatchedOp->getRHS()->IgnoreParenCasts()->getType()->isNullPtrType()) {
      return;
    }
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

void FunctionPointerEqualityCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "KeyedContainerTypes",
                utils::options::serializeStringList(KeyedContainerTypes));
}

} // namespace clang::tidy::misc
