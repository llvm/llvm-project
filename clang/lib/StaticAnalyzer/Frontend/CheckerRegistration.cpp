//===--- CheckerRegistration.cpp - Registration for the Analyzer Checkers -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include "clang/StaticAnalyzer/Frontend/FrontendActions.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace clang;
using namespace ento;
using llvm::sys::DynamicLibrary;

namespace {
class ClangCheckerRegistry : public CheckerRegistry {
  typedef void (*RegisterCheckersFn)(CheckerRegistry &);

  static bool isCompatibleAPIVersion(const char *versionString);
  static void warnIncompatible(DiagnosticsEngine *diags, StringRef pluginPath,
                               const char *pluginAPIVersion);

public:
  ClangCheckerRegistry(ArrayRef<std::string> plugins,
                       DiagnosticsEngine *diags = nullptr);
};

} // end anonymous namespace

ClangCheckerRegistry::ClangCheckerRegistry(ArrayRef<std::string> plugins,
                                           DiagnosticsEngine *diags) {
#define GET_CHECKERS
#define CHECKER(FULLNAME, CLASS, HELPTEXT)                                     \
  addChecker(register##CLASS, FULLNAME, HELPTEXT);
#include "clang/StaticAnalyzer/Checkers/Checkers.inc"
#undef CHECKER
#undef GET_CHECKERS

  for (ArrayRef<std::string>::iterator i = plugins.begin(), e = plugins.end();
       i != e; ++i) {
    // Get access to the plugin.
    std::string err;
    DynamicLibrary lib = DynamicLibrary::getPermanentLibrary(i->c_str(), &err);
    if (!lib.isValid()) {
      diags->Report(diag::err_fe_unable_to_load_plugin) << *i << err;
      continue;
    }

    // See if it's compatible with this build of clang.
    const char *pluginAPIVersion =
      (const char *) lib.getAddressOfSymbol("clang_analyzerAPIVersionString");
    if (!isCompatibleAPIVersion(pluginAPIVersion)) {
      warnIncompatible(diags, *i, pluginAPIVersion);
      continue;
    }

    // Register its checkers.
    RegisterCheckersFn registerPluginCheckers =
      (RegisterCheckersFn) (intptr_t) lib.getAddressOfSymbol(
                                                      "clang_registerCheckers");
    if (registerPluginCheckers)
      registerPluginCheckers(*this);
  }
}

bool ClangCheckerRegistry::isCompatibleAPIVersion(const char *versionString) {
  // If the version string is null, it's not an analyzer plugin.
  if (!versionString)
    return false;

  // For now, none of the static analyzer API is considered stable.
  // Versions must match exactly.
  return strcmp(versionString, CLANG_ANALYZER_API_VERSION_STRING) == 0;
}

void ClangCheckerRegistry::warnIncompatible(DiagnosticsEngine *diags,
                                            StringRef pluginPath,
                                            const char *pluginAPIVersion) {
  if (!diags)
    return;
  if (!pluginAPIVersion)
    return;

  diags->Report(diag::warn_incompatible_analyzer_plugin_api)
      << llvm::sys::path::filename(pluginPath);
  diags->Report(diag::note_incompatible_analyzer_plugin_api)
      << CLANG_ANALYZER_API_VERSION_STRING
      << pluginAPIVersion;
}

std::unique_ptr<CheckerManager> ento::createCheckerManager(
    ASTContext &context,
    AnalyzerOptions &opts,
    ArrayRef<std::string> plugins,
    ArrayRef<std::function<void(CheckerRegistry &)>> checkerRegistrationFns,
    DiagnosticsEngine &diags) {
  auto checkerMgr = llvm::make_unique<CheckerManager>(context, opts);

  ClangCheckerRegistry allCheckers(plugins, &diags);

  for (const auto &Fn : checkerRegistrationFns)
    Fn(allCheckers);

  allCheckers.initializeManager(*checkerMgr, opts, diags);
  allCheckers.validateCheckerOptions(opts, diags);
  checkerMgr->finishedCheckerRegistration();

  return checkerMgr;
}

void ento::printCheckerHelp(raw_ostream &out, ArrayRef<std::string> plugins) {
  out << "OVERVIEW: Clang Static Analyzer Checkers List\n\n";
  out << "USAGE: -analyzer-checker <CHECKER or PACKAGE,...>\n\n";

  ClangCheckerRegistry(plugins).printHelp(out);
}

void ento::printEnabledCheckerList(raw_ostream &out,
                                   ArrayRef<std::string> plugins,
                                   const AnalyzerOptions &opts,
                                   DiagnosticsEngine &diags) {
  out << "OVERVIEW: Clang Static Analyzer Enabled Checkers List\n\n";

  ClangCheckerRegistry(plugins).printList(out, opts, diags);
}

void ento::printAnalyzerConfigList(raw_ostream &out) {
  out << "OVERVIEW: Clang Static Analyzer -analyzer-config Option List\n\n";
  out << "USAGE: clang -cc1 [CLANG_OPTIONS] -analyzer-config "
                                        "<OPTION1=VALUE,OPTION2=VALUE,...>\n\n";
  out << "       clang -cc1 [CLANG_OPTIONS] -analyzer-config OPTION1=VALUE, "
                                      "-analyzer-config OPTION2=VALUE, ...\n\n";
  out << "       clang [CLANG_OPTIONS] -Xclang -analyzer-config -Xclang"
                                        "<OPTION1=VALUE,OPTION2=VALUE,...>\n\n";
  out << "       clang [CLANG_OPTIONS] -Xclang -analyzer-config -Xclang "
                              "OPTION1=VALUE, -Xclang -analyzer-config -Xclang "
                              "OPTION2=VALUE, ...\n\n";
  out << "OPTIONS:\n\n";

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

  constexpr size_t MinLineWidth = 70;
  constexpr size_t PadForOpt = 2;
  constexpr size_t OptionWidth = 30;
  constexpr size_t PadForDesc = PadForOpt + OptionWidth;
  static_assert(MinLineWidth > PadForDesc, "MinLineWidth must be greater!");

  llvm::formatted_raw_ostream FOut(out);

  for (const auto &Pair : PrintableOptions) {
    FOut.PadToColumn(PadForOpt) << Pair.first;

    // If the buffer's length is greater then PadForDesc, print a newline.
    if (FOut.getColumn() > PadForDesc)
      FOut << '\n';

    FOut.PadToColumn(PadForDesc);

    for (char C : Pair.second) {
      if (FOut.getColumn() > MinLineWidth && C == ' ') {
        FOut << '\n';
        FOut.PadToColumn(PadForDesc);
        continue;
      }
      FOut << C;
    }
    FOut << "\n\n";
  }
}
