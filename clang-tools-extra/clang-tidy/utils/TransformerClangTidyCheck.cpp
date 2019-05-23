//===---------- TransformerClangTidyCheck.cpp - clang-tidy ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TransformerClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace utils {
using tooling::RewriteRule;

void TransformerClangTidyCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
  Finder->addDynamicMatcher(tooling::detail::buildMatcher(Rule), this);
}

void TransformerClangTidyCheck::check(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (Result.Context->getDiagnostics().hasErrorOccurred())
    return;

  // Verify the existence and validity of the AST node that roots this rule.
  const ast_matchers::BoundNodes::IDToNodeMap &NodesMap = Result.Nodes.getMap();
  auto Root = NodesMap.find(RewriteRule::RootID);
  assert(Root != NodesMap.end() && "Transformation failed: missing root node.");
  SourceLocation RootLoc = Result.SourceManager->getExpansionLoc(
      Root->second.getSourceRange().getBegin());
  assert(RootLoc.isValid() && "Invalid location for Root node of match.");

  RewriteRule::Case Case = tooling::detail::findSelectedCase(Result, Rule);
  Expected<SmallVector<tooling::detail::Transformation, 1>> Transformations =
      tooling::detail::translateEdits(Result, Case.Edits);
  if (!Transformations) {
    llvm::errs() << "Rewrite failed: "
                 << llvm::toString(Transformations.takeError()) << "\n";
    return;
  }

  // No rewrite applied, but no error encountered either.
  if (Transformations->empty())
    return;

  StringRef Message = "no explanation";
  if (Case.Explanation) {
    if (Expected<std::string> E = Case.Explanation(Result))
      Message = *E;
    else
      llvm::errs() << "Error in explanation: " << llvm::toString(E.takeError())
                   << "\n";
  }
  DiagnosticBuilder Diag = diag(RootLoc, Message);
  for (const auto &T : *Transformations) {
    Diag << FixItHint::CreateReplacement(T.Range, T.Replacement);
  }
}

} // namespace utils
} // namespace tidy
} // namespace clang
