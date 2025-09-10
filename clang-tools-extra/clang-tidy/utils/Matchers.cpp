//===---------- Matchers.cpp - clang-tidy ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Matchers.h"
#include "ASTUtils.h"

namespace clang::tidy::matchers {

bool NotIdenticalStatementsPredicate::operator()(
    const clang::ast_matchers::internal::BoundNodesMap &Nodes) const {
  return !utils::areStatementsIdentical(Node.get<Stmt>(),
                                        Nodes.getNodeAs<Stmt>(ID), *Context);
}

MatchesAnyListedTypeNameMatcher::MatchesAnyListedTypeNameMatcher(
    llvm::ArrayRef<StringRef> NameList, bool CanonicalTypes)
    : NameMatchers(NameList.begin(), NameList.end()),
      CanonicalTypes(CanonicalTypes) {}

MatchesAnyListedTypeNameMatcher::~MatchesAnyListedTypeNameMatcher() = default;

bool MatchesAnyListedTypeNameMatcher::matches(
    const QualType &Node, ast_matchers::internal::ASTMatchFinder *Finder,
    ast_matchers::internal::BoundNodesTreeBuilder *Builder) const {

  if (NameMatchers.empty())
    return false;

  PrintingPolicy PrintingPolicyWithSuppressedTag(
      Finder->getASTContext().getLangOpts());
  PrintingPolicyWithSuppressedTag.PrintAsCanonical = CanonicalTypes;
  PrintingPolicyWithSuppressedTag.FullyQualifiedName = true;
  PrintingPolicyWithSuppressedTag.SuppressScope = false;
  PrintingPolicyWithSuppressedTag.SuppressTagKeyword = true;
  PrintingPolicyWithSuppressedTag.SuppressUnwrittenScope = true;
  std::string TypeName =
      Node.getUnqualifiedType().getAsString(PrintingPolicyWithSuppressedTag);

  return llvm::any_of(NameMatchers, [&TypeName](const llvm::Regex &NM) {
    return NM.isValid() && NM.match(TypeName);
  });
}

} // namespace clang::tidy::matchers
