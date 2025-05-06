//===--- Warnings.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// -Werror, ...
//
// Each warning option controls any number of actual warnings.
// Given a warning option 'foo', the following are valid:
//    -Wfoo, -Wno-foo, -Werror=foo
//
#include "clang/Basic/AllDiagnostics.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "llvm/ADT/StringRef.h"
#include <cstring>

namespace Fortran::frontend {

// EmitUnknownDiagWarning - Emit a warning and typo hint for unknown warning
// opts

static void EmitUnknownDiagWarning(clang::DiagnosticsEngine &diags,
                                   clang::diag::Flavor flavor,
                                   llvm::StringRef prefix,
                                   llvm::StringRef opt) {
  llvm::StringRef suggestion =
      clang::DiagnosticIDs::getNearestOption(flavor, opt);
  diags.Report(clang::diag::warn_unknown_diag_option)
      << (flavor == clang::diag::Flavor::WarningOrError ? 0 : 1)
      << (prefix.str() += std::string(opt)) << !suggestion.empty()
      << (prefix.str() += std::string(suggestion));
}

void processWarningOptions(clang::DiagnosticsEngine &diags,
                           const clang::DiagnosticOptions &opts) {
  diags.setIgnoreAllWarnings(opts.IgnoreWarnings);
  diags.setShowColors(opts.ShowColors);

  // If -pedantic or -pedantic-errors was specified, then we want to map all
  // extension diagnostics onto WARNING or ERROR.
  if (opts.PedanticErrors)
    diags.setExtensionHandlingBehavior(clang::diag::Severity::Error);
  else if (opts.Pedantic)
    diags.setExtensionHandlingBehavior(clang::diag::Severity::Warning);
  else
    diags.setExtensionHandlingBehavior(clang::diag::Severity::Ignored);

  llvm::SmallVector<clang::diag::kind, 10> _diags;
  const llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagIDs =
      diags.getDiagnosticIDs();
  for (unsigned i = 0, e = opts.Warnings.size(); i != e; ++i) {
    const auto flavor = clang::diag::Flavor::WarningOrError;
    llvm::StringRef opt = opts.Warnings[i];

    // Check to see if this warning starts with "no-", if so, this is a
    // negative form of the option.
    bool isPositive = !opt.consume_front("no-");

    // Figure out how this option affects the warning.  If -Wfoo, map the
    // diagnostic to a warning, if -Wno-foo, map it to ignore.
    clang::diag::Severity mapping = isPositive ? clang::diag::Severity::Warning
                                               : clang::diag::Severity::Ignored;

    // -Werror/-Wno-error is a special case, not controlled by the option table.
    // TODO: Adding support of "specifier" form of -Werror=foo.
    if (opt == "error") {
      diags.setWarningsAsErrors(isPositive);
      continue;
    }

    if (std::optional<clang::diag::Group> group =
            diagIDs->getGroupForWarningOption(opt)) {
      if (!diagIDs->isFlangWarningOption(group.value())) {
        // Warning option not supported by Flang
        // FIXME : Updating diagnostic error message when all warning options
        // will be supported
        const unsigned diagID =
            diags.getCustomDiagID(clang::DiagnosticsEngine::Error,
                                  "Warning option \"%0\" not supported.");
        diags.Report(diagID) << opt;
      }
    } else {
      // Unkown warning option.
      EmitUnknownDiagWarning(diags, flavor, isPositive ? "-W" : "-Wno-", opt);
    }
    diags.setSeverityForGroup(flavor, opt, mapping);
  }
}
} // namespace Fortran::frontend
