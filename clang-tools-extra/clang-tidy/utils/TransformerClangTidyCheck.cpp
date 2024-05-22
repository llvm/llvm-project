//===---------- TransformerClangTidyCheck.cpp - clang-tidy ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TransformerClangTidyCheck.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

namespace clang::tidy::utils {
using transformer::RewriteRuleWith;

#ifndef NDEBUG
static bool hasGenerator(const transformer::Generator<std::string> &G) {
  return G != nullptr;
}
#endif

static void verifyRule(const RewriteRuleWith<std::string> &Rule) {
  assert(llvm::all_of(Rule.Metadata, hasGenerator) &&
         "clang-tidy checks must have an explanation by default;"
         " explicitly provide an empty explanation if none is desired");
}

// If a string unintentionally containing '%' is passed as a diagnostic, Clang
// will claim the string is ill-formed and assert-fail. This function escapes
// such strings so they can be safely used in diagnostics.
std::string escapeForDiagnostic(std::string ToEscape) {
  // Optimize for the common case that the string does not contain `%` at the
  // cost of an extra scan over the string in the slow case.
  auto Pos = ToEscape.find('%');
  if (Pos == std::string::npos)
    return ToEscape;

  std::string Result;
  Result.reserve(ToEscape.size());
  // Convert position to a count.
  ++Pos;
  Result.append(ToEscape, 0, Pos);
  Result += '%';

  for (auto N = ToEscape.size(); Pos < N; ++Pos) {
    const char C = ToEscape.at(Pos);
    Result += C;
    if (C == '%')
      Result += '%';
  }

  return Result;
}

TransformerClangTidyCheck::TransformerClangTidyCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Inserter(Options.getLocalOrGlobal("IncludeStyle", IncludeSorter::IS_LLVM),
               areDiagsSelfContained()) {}

// This constructor cannot dispatch to the simpler one (below), because, in
// order to get meaningful results from `getLangOpts` and `Options`, we need the
// `ClangTidyCheck()` constructor to have been called. If we were to dispatch,
// we would be accessing `getLangOpts` and `Options` before the underlying
// `ClangTidyCheck` instance was properly initialized.
TransformerClangTidyCheck::TransformerClangTidyCheck(
    std::function<std::optional<RewriteRuleWith<std::string>>(
        const LangOptions &, const OptionsView &)>
        MakeRule,
    StringRef Name, ClangTidyContext *Context)
    : TransformerClangTidyCheck(Name, Context) {
  if (std::optional<RewriteRuleWith<std::string>> R =
          MakeRule(getLangOpts(), Options))
    setRule(std::move(*R));
}

TransformerClangTidyCheck::TransformerClangTidyCheck(
    RewriteRuleWith<std::string> R, StringRef Name, ClangTidyContext *Context)
    : TransformerClangTidyCheck(Name, Context) {
  setRule(std::move(R));
}

void TransformerClangTidyCheck::setRule(
    transformer::RewriteRuleWith<std::string> R) {
  verifyRule(R);
  Rule = std::move(R);
}

void TransformerClangTidyCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void TransformerClangTidyCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
  if (!Rule.Cases.empty())
    for (auto &Matcher : transformer::detail::buildMatchers(Rule))
      Finder->addDynamicMatcher(Matcher, this);
}

void TransformerClangTidyCheck::check(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (Result.Context->getDiagnostics().hasErrorOccurred())
    return;

  size_t I = transformer::detail::findSelectedCase(Result, Rule);
  Expected<SmallVector<transformer::Edit, 1>> Edits =
      Rule.Cases[I].Edits(Result);
  if (!Edits) {
    llvm::errs() << "Rewrite failed: " << llvm::toString(Edits.takeError())
                 << "\n";
    return;
  }

  // No rewrite applied, but no error encountered either.
  if (Edits->empty())
    return;

  Expected<std::string> Explanation = Rule.Metadata[I]->eval(Result);
  if (!Explanation) {
    llvm::errs() << "Error in explanation: "
                 << llvm::toString(Explanation.takeError()) << "\n";
    return;
  }

  // Associate the diagnostic with the location of the first change.
  {
    DiagnosticBuilder Diag =
        diag((*Edits)[0].Range.getBegin(), escapeForDiagnostic(*Explanation));
    for (const auto &T : *Edits) {
      switch (T.Kind) {
      case transformer::EditKind::Range:
        Diag << FixItHint::CreateReplacement(T.Range, T.Replacement);
        break;
      case transformer::EditKind::AddInclude:
        Diag << Inserter.createIncludeInsertion(
            Result.SourceManager->getFileID(T.Range.getBegin()), T.Replacement);
        break;
      }
    }
  }
  // Emit potential notes.
  for (const auto &T : *Edits) {
    if (!T.Note.empty()) {
      diag(T.Range.getBegin(), escapeForDiagnostic(T.Note),
           DiagnosticIDs::Note);
    }
  }
}

void TransformerClangTidyCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", Inserter.getStyle());
}

} // namespace clang::tidy::utils
