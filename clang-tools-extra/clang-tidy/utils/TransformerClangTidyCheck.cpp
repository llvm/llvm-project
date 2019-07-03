//===---------- TransformerClangTidyCheck.cpp - clang-tidy ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TransformerClangTidyCheck.h"
#include "llvm/ADT/STLExtras.h"

namespace clang {
namespace tidy {
namespace utils {
using tooling::RewriteRule;

#ifndef NDEBUG
static bool hasExplanation(const RewriteRule::Case &C) {
  return C.Explanation != nullptr;
}
#endif

// This constructor cannot dispatch to the simpler one (below), because, in
// order to get meaningful results from `getLangOpts` and `Options`, we need the
// `ClangTidyCheck()` constructor to have been called. If we were to dispatch,
// we would be accessing `getLangOpts` and `Options` before the underlying
// `ClangTidyCheck` instance was properly initialized.
TransformerClangTidyCheck::TransformerClangTidyCheck(
    std::function<Optional<RewriteRule>(const LangOptions &,
                                        const OptionsView &)>
        MakeRule,
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context), Rule(MakeRule(getLangOpts(), Options)) {
  if (Rule)
    assert(llvm::all_of(Rule->Cases, hasExplanation) &&
           "clang-tidy checks must have an explanation by default;"
           " explicitly provide an empty explanation if none is desired");
}

TransformerClangTidyCheck::TransformerClangTidyCheck(RewriteRule R,
                                                     StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context), Rule(std::move(R)) {
  assert(llvm::all_of(Rule->Cases, hasExplanation) &&
         "clang-tidy checks must have an explanation by default;"
         " explicitly provide an empty explanation if none is desired");
}

void TransformerClangTidyCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  // Only allocate and register the IncludeInsert when some `Case` will add
  // includes.
  if (Rule && llvm::any_of(Rule->Cases, [](const RewriteRule::Case &C) {
        return !C.AddedIncludes.empty();
      })) {
    Inserter = llvm::make_unique<IncludeInserter>(
        SM, getLangOpts(), utils::IncludeSorter::IS_LLVM);
    PP->addPPCallbacks(Inserter->CreatePPCallbacks());
  }
}

void TransformerClangTidyCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
  if (Rule)
    Finder->addDynamicMatcher(tooling::detail::buildMatcher(*Rule), this);
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

  assert(Rule && "check() should not fire if Rule is None");
  RewriteRule::Case Case = tooling::detail::findSelectedCase(Result, *Rule);
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

  Expected<std::string> Explanation = Case.Explanation(Result);
  if (!Explanation) {
    llvm::errs() << "Error in explanation: "
                 << llvm::toString(Explanation.takeError()) << "\n";
    return;
  }
  DiagnosticBuilder Diag = diag(RootLoc, *Explanation);
  for (const auto &T : *Transformations) {
    Diag << FixItHint::CreateReplacement(T.Range, T.Replacement);
  }

  for (const auto &I : Case.AddedIncludes) {
    auto &Header = I.first;
    if (Optional<FixItHint> Fix = Inserter->CreateIncludeInsertion(
            Result.SourceManager->getMainFileID(), Header,
            /*IsAngled=*/I.second == tooling::IncludeFormat::Angled)) {
      Diag << *Fix;
    }
  }
}

} // namespace utils
} // namespace tidy
} // namespace clang
