//===--- FlangTidyContext.h - flang-tidy ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_TOOLS_FLANG_TIDY_FLANGTIDYCONTEXT_H
#define LLVM_FLANG_TOOLS_FLANG_TIDY_FLANGTIDYCONTEXT_H

#include "FlangTidyOptions.h"
#include "flang/Semantics/semantics.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include <clang/Basic/Diagnostic.h>

namespace Fortran::tidy {

/// This class is used to manage the context for Flang Tidy checks.
/// It contains the enabled checks and the semantics context.
/// It provides methods to check if a specific check is enabled and to access
/// the semantics context.
///
/// For user-facing documentation, see:
/// https://flang.llvm.org/@PLACEHOLDER@/flang-tidy.html
class FlangTidyContext {
public:
  FlangTidyContext(const FlangTidyOptions &options,
                   semantics::SemanticsContext *ctx) {
    Options = options;
    for (const auto &CheckName : options.enabledChecks) {
      Checks.insert(CheckName);
    }
    for (const auto &CheckName : options.enabledWarningsAsErrors) {
      WarningsAsErrors.insert(CheckName);
    }
    Context = ctx;
  }

  bool isEnabled(const llvm::StringRef &CheckName,
                 llvm::SmallSet<llvm::StringRef, 16> const &Set) const {
    bool enabled = false;
    for (const auto &Pattern : Set) {
      if (Pattern.starts_with("-")) {
        llvm::StringRef DisablePrefix = Pattern.drop_front(1);
        if (DisablePrefix.ends_with("*")) {
          DisablePrefix = DisablePrefix.drop_back(1);
          if (CheckName.starts_with(DisablePrefix)) {
            enabled = false;
          }
        } else if (DisablePrefix == CheckName) {
          enabled = false;
        }
      } else if (Pattern.ends_with("*")) {
        llvm::StringRef EnablePrefix = Pattern.drop_back(1);
        if (CheckName.starts_with(EnablePrefix)) {
          enabled = true;
        }
      } else if (Pattern == CheckName) {
        enabled = true;
      }
    }
    return enabled;
  }

  bool isCheckEnabled(const llvm::StringRef &CheckName) const {
    return isEnabled(CheckName, Checks);
  }

  bool isWarningsAsErrorsEnabled(const llvm::StringRef &CheckName) const {
    return isEnabled(CheckName, WarningsAsErrors);
  }

  semantics::SemanticsContext &getSemanticsContext() const { return *Context; }

  /// Get the FlangTidy options
  const FlangTidyOptions &getOptions() const { return Options; }

public:
  /// List of enabled checks.
  llvm::SmallSet<llvm::StringRef, 16> Checks;
  /// List of checks for which to turn warnings into errors.
  llvm::SmallSet<llvm::StringRef, 16> WarningsAsErrors;
  /// The semantics context used for the checks.
  semantics::SemanticsContext *Context;
  FlangTidyOptions Options;
};

} // namespace Fortran::tidy

#endif // LLVM_FLANG_TOOLS_FLANG_TIDY_FLANGTIDYCONTEXT_H
