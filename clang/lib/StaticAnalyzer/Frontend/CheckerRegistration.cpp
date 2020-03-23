//===--- CheckerRegistration.cpp - Registration for the Analyzer Checkers -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the registration function for the analyzer checkers.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Frontend/CheckerRegistration.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/StaticAnalyzer/Core/AnalyzerOptions.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include "clang/StaticAnalyzer/Frontend/FrontendActions.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace clang;
using namespace ento;

std::unique_ptr<CheckerManager> ento::createCheckerManager(
    ASTContext &context,
    AnalyzerOptions &opts,
    ArrayRef<std::string> plugins,
    ArrayRef<std::function<void(CheckerRegistry &)>> checkerRegistrationFns,
    DiagnosticsEngine &diags) {
  auto checkerMgr = std::make_unique<CheckerManager>(context, opts);

  CheckerRegistry allCheckers(plugins, diags, opts, context.getLangOpts(),
                              checkerRegistrationFns);

  allCheckers.initializeManager(*checkerMgr);
  allCheckers.validateCheckerOptions();
  checkerMgr->finishedCheckerRegistration();

  return checkerMgr;
}

void ento::printCheckerHelp(raw_ostream &out, ArrayRef<std::string> plugins,
                            AnalyzerOptions &anopts,
                            DiagnosticsEngine &diags,
                            const LangOptions &langOpts) {
  out << "OVERVIEW: Clang Static Analyzer Checkers List\n\n";
  out << "USAGE: -analyzer-checker <CHECKER or PACKAGE,...>\n\n";

  CheckerRegistry(plugins, diags, anopts, langOpts)
      .printCheckerWithDescList(out);
}

void ento::printEnabledCheckerList(raw_ostream &out,
                                   ArrayRef<std::string> plugins,
                                   AnalyzerOptions &anopts,
                                   DiagnosticsEngine &diags,
                                   const LangOptions &langOpts) {
  out << "OVERVIEW: Clang Static Analyzer Enabled Checkers List\n\n";

  CheckerRegistry(plugins, diags, anopts, langOpts)
      .printEnabledCheckerList(out);
}

void ento::printCheckerConfigList(raw_ostream &OS,
                                  ArrayRef<std::string> plugins,
                                  AnalyzerOptions &opts,
                                  DiagnosticsEngine &diags,
                                  const LangOptions &LangOpts) {
  CheckerRegistry(plugins, diags, opts, LangOpts)
      .printCheckerOptionList(OS);
}

void ento::printAnalyzerConfigList(raw_ostream &out) {
  // FIXME: This message sounds scary, should be scary, but incorrectly states
  // that all configs are super dangerous. In reality, many of them should be
  // accessible to the user. We should create a user-facing subset of config
  // options under a different frontend flag.
  out << R"(
OVERVIEW: Clang Static Analyzer -analyzer-config Option List

The following list of configurations are meant for development purposes only, as
some of the variables they define are set to result in the most optimal
analysis. Setting them to other values may drastically change how the analyzer
behaves, and may even result in instabilities, crashes!

USAGE: -analyzer-config <OPTION1=VALUE,OPTION2=VALUE,...>
       -analyzer-config OPTION1=VALUE, -analyzer-config OPTION2=VALUE, ...
OPTIONS:
)";

  using OptionAndDescriptionTy = std::pair<StringRef, std::string>;
  OptionAndDescriptionTy PrintableOptions[] = {
#define ANALYZER_OPTION(TYPE, NAME, CMDFLAG, DESC, DEFAULT_VAL)                \
    {                                                                          \
      CMDFLAG,                                                                 \
      llvm::Twine(llvm::Twine() + "(" +                                        \
                  (StringRef(#TYPE) == "StringRef" ? "string" : #TYPE ) +      \
                  ") " DESC                                                    \
                  " (default: " #DEFAULT_VAL ")").str()                        \
    },

#define ANALYZER_OPTION_DEPENDS_ON_USER_MODE(TYPE, NAME, CMDFLAG, DESC,        \
                                             SHALLOW_VAL, DEEP_VAL)            \
    {                                                                          \
      CMDFLAG,                                                                 \
      llvm::Twine(llvm::Twine() + "(" +                                        \
                  (StringRef(#TYPE) == "StringRef" ? "string" : #TYPE ) +      \
                  ") " DESC                                                    \
                  " (default: " #SHALLOW_VAL " in shallow mode, " #DEEP_VAL    \
                  " in deep mode)").str()                                      \
    },
#include "clang/StaticAnalyzer/Core/AnalyzerOptions.def"
#undef ANALYZER_OPTION
#undef ANALYZER_OPTION_DEPENDS_ON_USER_MODE
  };

  llvm::sort(PrintableOptions, [](const OptionAndDescriptionTy &LHS,
                                  const OptionAndDescriptionTy &RHS) {
    return LHS.first < RHS.first;
  });

  for (const auto &Pair : PrintableOptions) {
    AnalyzerOptions::printFormattedEntry(out, Pair, /*InitialPad*/ 2,
                                         /*EntryWidth*/ 30,
                                         /*MinLineWidth*/ 70);
    out << "\n\n";
  }
}
