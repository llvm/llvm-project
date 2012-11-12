//===--- Warnings.cpp - C-Language Front-end ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Command line warning options handler.
//
//===----------------------------------------------------------------------===//
//
// This file is responsible for handling all warning options. This includes
// a number of -Wfoo options and their variants, which are driven by TableGen-
// generated data, and the special cases -pedantic, -pedantic-errors, -w,
// -Werror and -Wfatal-errors.
//
// Each warning option controls any number of actual warnings.
// Given a warning option 'foo', the following are valid:
//    -Wfoo, -Wno-foo, -Werror=foo, -Wfatal-errors=foo
//
#include "clang/Frontend/Utils.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include <cstring>
#include <utility>
#include <algorithm>
using namespace clang;

// EmitUnknownDiagWarning - Emit a warning and typo hint for unknown warning
// opts
static void EmitUnknownDiagWarning(DiagnosticsEngine &Diags,
                                  StringRef Prefix, StringRef Opt,
                                  bool isPositive) {
  StringRef Suggestion = DiagnosticIDs::getNearestWarningOption(Opt);
  if (!Suggestion.empty())
    Diags.Report(isPositive? diag::warn_unknown_warning_option_suggest :
                             diag::warn_unknown_negative_warning_option_suggest)
      << (Prefix.str() += Opt) << (Prefix.str() += Suggestion);
  else
    Diags.Report(isPositive? diag::warn_unknown_warning_option :
                             diag::warn_unknown_negative_warning_option)
      << (Prefix.str() += Opt);
}

void clang::ProcessWarningOptions(DiagnosticsEngine &Diags,
                                  const DiagnosticOptions &Opts) {
  Diags.setSuppressSystemWarnings(true);  // Default to -Wno-system-headers
  Diags.setIgnoreAllWarnings(Opts.IgnoreWarnings);
  Diags.setShowOverloads(Opts.getShowOverloads());

  Diags.setElideType(Opts.ElideType);
  Diags.setPrintTemplateTree(Opts.ShowTemplateTree);
  Diags.setShowColors(Opts.ShowColors);
 
  // Handle -ferror-limit
  if (Opts.ErrorLimit)
    Diags.setErrorLimit(Opts.ErrorLimit);
  if (Opts.TemplateBacktraceLimit)
    Diags.setTemplateBacktraceLimit(Opts.TemplateBacktraceLimit);
  if (Opts.ConstexprBacktraceLimit)
    Diags.setConstexprBacktraceLimit(Opts.ConstexprBacktraceLimit);

  // If -pedantic or -pedantic-errors was specified, then we want to map all
  // extension diagnostics onto WARNING or ERROR unless the user has futz'd
  // around with them explicitly.
  if (Opts.PedanticErrors)
    Diags.setExtensionHandlingBehavior(DiagnosticsEngine::Ext_Error);
  else if (Opts.Pedantic)
    Diags.setExtensionHandlingBehavior(DiagnosticsEngine::Ext_Warn);
  else
    Diags.setExtensionHandlingBehavior(DiagnosticsEngine::Ext_Ignore);

  llvm::SmallVector<diag::kind, 10> _Diags;
  const IntrusiveRefCntPtr< DiagnosticIDs > DiagIDs =
    Diags.getDiagnosticIDs();
  // We parse the warning options twice.  The first pass sets diagnostic state,
  // while the second pass reports warnings/errors.  This has the effect that
  // we follow the more canonical "last option wins" paradigm when there are 
  // conflicting options.
  for (unsigned Report = 0, ReportEnd = 2; Report != ReportEnd; ++Report) {
    bool SetDiagnostic = (Report == 0);
    for (unsigned i = 0, e = Opts.Warnings.size(); i != e; ++i) {
      StringRef Opt = Opts.Warnings[i];
      StringRef OrigOpt = Opts.Warnings[i];

      // Treat -Wformat=0 as an alias for -Wno-format.
      if (Opt == "format=0")
        Opt = "no-format";

      // Check to see if this warning starts with "no-", if so, this is a
      // negative form of the option.
      bool isPositive = true;
      if (Opt.startswith("no-")) {
        isPositive = false;
        Opt = Opt.substr(3);
      }

      // Figure out how this option affects the warning.  If -Wfoo, map the
      // diagnostic to a warning, if -Wno-foo, map it to ignore.
      diag::Mapping Mapping = isPositive ? diag::MAP_WARNING : diag::MAP_IGNORE;
      
      // -Wsystem-headers is a special case, not driven by the option table.  It
      // cannot be controlled with -Werror.
      if (Opt == "system-headers") {
        if (SetDiagnostic)
          Diags.setSuppressSystemWarnings(!isPositive);
        continue;
      }
      
      // -Weverything is a special case as well.  It implicitly enables all
      // warnings, including ones not explicitly in a warning group.
      if (Opt == "everything") {
        if (SetDiagnostic) {
          if (isPositive) {
            Diags.setEnableAllWarnings(true);
          } else {
            Diags.setEnableAllWarnings(false);
            Diags.setMappingToAllDiagnostics(diag::MAP_IGNORE);
          }
        }
        continue;
      }
      
      // -Werror/-Wno-error is a special case, not controlled by the option 
      // table. It also has the "specifier" form of -Werror=foo and -Werror-foo.
      if (Opt.startswith("error")) {
        StringRef Specifier;
        if (Opt.size() > 5) {  // Specifier must be present.
          if ((Opt[5] != '=' && Opt[5] != '-') || Opt.size() == 6) {
            if (Report)
              Diags.Report(diag::warn_unknown_warning_specifier)
                << "-Werror" << ("-W" + OrigOpt.str());
            continue;
          }
          Specifier = Opt.substr(6);
        }
        
        if (Specifier.empty()) {
          if (SetDiagnostic)
            Diags.setWarningsAsErrors(isPositive);
          continue;
        }
        
        if (SetDiagnostic) {
          // Set the warning as error flag for this specifier.
          Diags.setDiagnosticGroupWarningAsError(Specifier, isPositive);
        } else if (DiagIDs->getDiagnosticsInGroup(Specifier, _Diags)) {
          EmitUnknownDiagWarning(Diags, "-Werror=", Specifier, isPositive);
        }
        continue;
      }
      
      // -Wfatal-errors is yet another special case.
      if (Opt.startswith("fatal-errors")) {
        StringRef Specifier;
        if (Opt.size() != 12) {
          if ((Opt[12] != '=' && Opt[12] != '-') || Opt.size() == 13) {
            if (Report)
              Diags.Report(diag::warn_unknown_warning_specifier)
                << "-Wfatal-errors" << ("-W" + OrigOpt.str());
            continue;
          }
          Specifier = Opt.substr(13);
        }

        if (Specifier.empty()) {
          if (SetDiagnostic)
            Diags.setErrorsAsFatal(isPositive);
          continue;
        }
        
        if (SetDiagnostic) {
          // Set the error as fatal flag for this specifier.
          Diags.setDiagnosticGroupErrorAsFatal(Specifier, isPositive);
        } else if (DiagIDs->getDiagnosticsInGroup(Specifier, _Diags)) {
          EmitUnknownDiagWarning(Diags, "-Wfatal-errors=", Specifier,
                                 isPositive);
        }
        continue;
      }
      
      if (Report) {
        if (DiagIDs->getDiagnosticsInGroup(Opt, _Diags))
          EmitUnknownDiagWarning(Diags, isPositive ? "-W" : "-Wno-", Opt,
                                 isPositive);
      } else {
        Diags.setDiagnosticGroupMapping(Opt, Mapping);
      }
    }
  }
}
