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
using transformer::RewriteRule;

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
    : ClangTidyCheck(Name, Context), Rule(MakeRule(getLangOpts(), Options)),
      IncludeStyle(Options.getLocalOrGlobal("IncludeStyle",
                                            IncludeSorter::getMapping(),
                                            IncludeSorter::IS_LLVM)) {
  if (Rule)
    assert(llvm::all_of(Rule->Cases, hasExplanation) &&
           "clang-tidy checks must have an explanation by default;"
           " explicitly provide an empty explanation if none is desired");
}

TransformerClangTidyCheck::TransformerClangTidyCheck(RewriteRule R,
                                                     StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context), Rule(std::move(R)),
      IncludeStyle(Options.getLocalOrGlobal("IncludeStyle",
                                            IncludeSorter::getMapping(),
                                            IncludeSorter::IS_LLVM)) {
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
    Inserter =
        std::make_unique<IncludeInserter>(SM, getLangOpts(), IncludeStyle);
    PP->addPPCallbacks(Inserter->CreatePPCallbacks());
  }
}

void TransformerClangTidyCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
  if (Rule)
    for (auto &Matcher : transformer::detail::buildMatchers(*Rule))
      Finder->addDynamicMatcher(Matcher, this);
}

void TransformerClangTidyCheck::check(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (Result.Context->getDiagnostics().hasErrorOccurred())
    return;

  assert(Rule && "check() should not fire if Rule is None");
  RewriteRule::Case Case = transformer::detail::findSelectedCase(Result, *Rule);
  Expected<SmallVector<transformer::Edit, 1>> Edits = Case.Edits(Result);
  if (!Edits) {
    llvm::errs() << "Rewrite failed: " << llvm::toString(Edits.takeError())
                 << "\n";
    return;
  }

  // No rewrite applied, but no error encountered either.
  if (Edits->empty())
    return;

  Expected<std::string> Explanation = Case.Explanation->eval(Result);
  if (!Explanation) {
    llvm::errs() << "Error in explanation: "
                 << llvm::toString(Explanation.takeError()) << "\n";
    return;
  }

  // Associate the diagnostic with the location of the first change.
  DiagnosticBuilder Diag = diag((*Edits)[0].Range.getBegin(), *Explanation);
  for (const auto &T : *Edits)
    Diag << FixItHint::CreateReplacement(T.Range, T.Replacement);

  for (const auto &I : Case.AddedIncludes) {
    auto &Header = I.first;
    if (Optional<FixItHint> Fix = Inserter->CreateIncludeInsertion(
            Result.SourceManager->getMainFileID(), Header,
            /*IsAngled=*/I.second == transformer::IncludeFormat::Angled)) {
      Diag << *Fix;
    }
  }
}

void TransformerClangTidyCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", IncludeStyle,
                IncludeSorter::getMapping());
}

} // namespace utils
} // namespace tidy
} // namespace clang
