//===- BoundsSafetyArgs.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/Driver/BoundsSafetyArgs.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Driver/Options.h"

using namespace llvm::opt;
using namespace clang::driver::options;

namespace clang {

namespace driver {

LangOptions::BoundsSafetyNewChecksMaskIntTy
ParseBoundsSafetyNewChecksMaskFromArgs(const llvm::opt::ArgList &Args,
                                       DiagnosticsEngine *Diags) {
  auto FilteredArgs =
      Args.filtered(OPT_fbounds_safety_bringup_missing_checks_EQ,
                    OPT_fno_bounds_safety_bringup_missing_checks_EQ);
  if (FilteredArgs.empty()) {
    // No flags present. Use the default
    return LangOptions::getDefaultBoundsSafetyNewChecksMask();
  }

  // If flags are present then start with BS_CHK_None as the initial mask and
  // update the mask based on the flags. This preserves compiler behavior for
  // users that adopted the `-fbounds-safety-bringup-missing-checks` flag when
  // `getDefaultBoundsSafetyNewChecksMask() == BS_CHK_None`.
  LangOptions::BoundsSafetyNewChecksMaskIntTy Result = LangOptions::BS_CHK_None;
  // All the options will be applied as masks in the command line order, to make
  // it easier to enable all but certain checks (or disable all but certain
  // checks).
  for (const Arg *A : FilteredArgs) {
    for (const char *Value : A->getValues()) {
      std::optional<LangOptions::BoundsSafetyNewChecksMaskIntTy> Mask =
          llvm::StringSwitch<
              std::optional<LangOptions::BoundsSafetyNewChecksMaskIntTy>>(Value)
              .Case("access_size", LangOptions::BS_CHK_AccessSize)
              .Case("indirect_count_update",
                    LangOptions::BS_CHK_IndirectCountUpdate)
              .Case("return_size", LangOptions::BS_CHK_ReturnSize)
              .Case("ended_by_lower_bound",
                    LangOptions::BS_CHK_EndedByLowerBound)
              .Case("compound_literal_init",
                    LangOptions::BS_CHK_CompoundLiteralInit)
              .Case("libc_attributes", LangOptions::BS_CHK_LibCAttributes)
              .Case("array_subscript_agg",
                    LangOptions::BS_CHK_ArraySubscriptAgg)
              .Case("all",
                    LangOptions::getBoundsSafetyNewChecksMaskForGroup("all"))
              .Case(
                  "batch_0",
                  LangOptions::getBoundsSafetyNewChecksMaskForGroup("batch_0"))
              .Case("none", LangOptions::BS_CHK_None)
              .Default(std::nullopt);
      if (!Mask) {
        if (Diags)
          Diags->Report(diag::err_drv_invalid_value)
              << A->getSpelling() << Value;
        break;
      }

      bool IsPosFlag =
          A->getOption().matches(OPT_fbounds_safety_bringup_missing_checks_EQ)
              ? true
              : false;

      // `-fbounds-safety-bringup-missing-checks=none` would do nothing as the
      // masks are additive, which is unlikely to be intended. To disable all
      // checks, `-fno-bounds-safety-bringup-missing-checks=all` should be used
      // instead. Hence, "none" is not supported, triggering an error with the
      // suggestion.
      if (*Mask == LangOptions::BS_CHK_None) {
        if (Diags)
          Diags->Report(diag::err_drv_invalid_value_with_flag_suggestion)
              << A->getSpelling() << Value
              << (IsPosFlag ? "-fno-bounds-safety-bringup-missing-checks"
                            : "-fbounds-safety-bringup-missing-checks");
        break;
      }

      if (IsPosFlag) {
        Result |= *Mask;
      } else {
        assert(A->getOption().matches(
            OPT_fno_bounds_safety_bringup_missing_checks_EQ));
        Result &= ~(*Mask);
      }
    }
  }
  return Result;
}

} // namespace driver

} // namespace clang
