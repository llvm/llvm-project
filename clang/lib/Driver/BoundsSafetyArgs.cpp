//===- BoundsSafetyArgs.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/Driver/BoundsSafetyArgs.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Driver/Options.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/bit.h"
#include <array>

using namespace llvm::opt;
using namespace clang::driver::options;

namespace clang {

namespace driver {

static void DiagnoseDisabledBoundsSafetyChecks(
    LangOptions::BoundsSafetyNewChecksMaskIntTy EnabledChecks,
    DiagnosticsEngine *Diags,
    LangOptions::BoundsSafetyNewChecksMaskIntTy
        IndividualChecksExplicitlyDisabled) {
  struct BoundsCheckBatch {
    const char *Name;
    LangOptionsBase::BoundsSafetyNewChecksMaskIntTy Checks;
  };

  // Batches of checks should be ordered with newest first
  std::array<BoundsCheckBatch, 2> Batches = {
      {// We deliberately don't include `all` here because that batch
       // isn't stable over time (unlike batches like `batch_0`) so we
       // don't want to suggest users start using it.
       {"batch_0",
        LangOptions::getBoundsSafetyNewChecksMaskForGroup("batch_0")},
       {"none", LangOptions::BS_CHK_None}}};

  LangOptionsBase::BoundsSafetyNewChecksMaskIntTy DiagnosedDisabledChecks =
      LangOptions::BS_CHK_None;

  // Loop over all batches except "none"
  for (size_t BatchIdx = 0; BatchIdx < Batches.size() - 1; ++BatchIdx) {
    auto ChecksInCurrentBatch = Batches[BatchIdx].Checks;
    auto ChecksInOlderBatch = Batches[BatchIdx + 1].Checks;
    auto ChecksInCurrentBatchOnly = ChecksInCurrentBatch & ~ChecksInOlderBatch;
    const auto *CurrentBatchName = Batches[BatchIdx].Name;

    if ((EnabledChecks & ChecksInCurrentBatchOnly) == ChecksInCurrentBatchOnly)
      continue; // All checks in batch are enabled. Nothing to diagnose.

    // Diagnose disabled bounds checks

    if ((EnabledChecks & ChecksInCurrentBatchOnly) == 0) {
      // None of the checks in the current batch are enabled. Diagnose this
      // once for all the checks in the batch.
      if ((DiagnosedDisabledChecks & ChecksInCurrentBatchOnly) !=
          ChecksInCurrentBatchOnly) {
        Diags->Report(diag::warn_bounds_safety_new_checks_none)
            << CurrentBatchName;
        DiagnosedDisabledChecks |= ChecksInCurrentBatchOnly;
      }
      continue;
    }

    // Some (but not all) checks are disabled in the current batch. Iterate over
    // each check in the batch and emit a diagnostic for each disabled check
    // in the batch.
    assert(ChecksInCurrentBatch > 0);
    auto FirstCheckInBatch = 1 << llvm::countr_zero(ChecksInCurrentBatch);
    for (size_t CheckBit = FirstCheckInBatch;
         CheckBit <= LangOptions::BS_CHK_MaxMask; CheckBit <<= 1) {
      assert(CheckBit != 0);
      if ((CheckBit & ChecksInCurrentBatch) == 0)
        continue; // Check not in batch

      if (EnabledChecks & CheckBit)
        continue; // Check is active

      // Diagnose disabled check
      if (!(DiagnosedDisabledChecks & CheckBit)) {
        size_t CheckNumber = llvm::countr_zero(CheckBit);
        // If we always suggested enabling the current batch that
        // could be confusing if the user passed something like
        // `-fbounds-safety-bringup-missing-checks=batch_0
        // -fno-bounds-safety-bringup-missing-checks=access_size`. To avoid
        // this we detect when the check corresponding to `CheckBit` has been
        // explicitly disabled on the command line and in that case we suggeset
        // removing the flag.
        bool SuggestRemovingFlag =
            CheckBit & IndividualChecksExplicitlyDisabled;
        Diags->Report(diag::warn_bounds_safety_new_checks_mixed)
            << CheckNumber << SuggestRemovingFlag << CurrentBatchName;
        DiagnosedDisabledChecks |= CheckBit;
      }
    }
  }
}

LangOptions::BoundsSafetyNewChecksMaskIntTy
ParseBoundsSafetyNewChecksMaskFromArgs(const llvm::opt::ArgList &Args,
                                       DiagnosticsEngine *Diags,
                                       bool DiagnoseMissingChecks) {
  assert((!DiagnoseMissingChecks || Diags) &&
         "Cannot diagnose missing checks when Diags is a nullptr");
  LangOptions::BoundsSafetyNewChecksMaskIntTy
      IndividualChecksExplicitlyDisabled = LangOptions::BS_CHK_None;
  auto FilteredArgs =
      Args.filtered(OPT_fbounds_safety_bringup_missing_checks_EQ,
                    OPT_fno_bounds_safety_bringup_missing_checks_EQ);
  if (FilteredArgs.empty()) {
    // No flags present. Use the default
    auto Result = LangOptions::getDefaultBoundsSafetyNewChecksMask();
    DiagnoseDisabledBoundsSafetyChecks(Result, Diags,
                                       IndividualChecksExplicitlyDisabled);
    return Result;
  }

  // If flags are present then start with BS_CHK_None as the initial mask and
  // update the mask based on the flags. This preserves compiler behavior for
  // users that adopted the `-fbounds-safety-bringup-missing-checks` flag when
  // `getDefaultBoundsSafetyNewChecksMask() == BS_CHK_None`.
  LangOptions::BoundsSafetyNewChecksMaskIntTy Result = LangOptions::BS_CHK_None;
  // All the options will be applied as masks in the command line order, to make
  // it easier to enable all but certain checks (or disable all but certain
  // checks).
  const auto Batch0Checks =
      LangOptions::getBoundsSafetyNewChecksMaskForGroup("batch_0");
  const auto AllChecks =
      LangOptions::getBoundsSafetyNewChecksMaskForGroup("all");
  bool Errors = false;
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
              .Case("all", AllChecks)
              .Case("batch_0", Batch0Checks)
              .Case("none", LangOptions::BS_CHK_None)
              .Default(std::nullopt);

      if (!Mask) {
        if (Diags)
          Diags->Report(diag::err_drv_invalid_value)
              << A->getSpelling() << Value;
        Errors = true;
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
        Errors = true;
        break;
      }

      if (IsPosFlag) {
        Result |= *Mask;
      } else {
        assert(A->getOption().matches(
            OPT_fno_bounds_safety_bringup_missing_checks_EQ));
        Result &= ~(*Mask);
      }

      // Update which checks have been explicitly disabled. E.g.
      // `-fno-bounds-safety-bringup-missing-checks=access_size`.
      if (llvm::has_single_bit(*Mask)) {
        // A single check was enabled/disabled
        if (IsPosFlag)
          IndividualChecksExplicitlyDisabled &= ~(*Mask);
        else
          IndividualChecksExplicitlyDisabled |= *Mask;
      } else {
        // A batch of checks were enabled/disabled. Any checks in that batch
        // are no longer explicitly set.
        IndividualChecksExplicitlyDisabled &= ~(*Mask);
      }
    }
  }
  if (DiagnoseMissingChecks && Diags && !Errors)
    DiagnoseDisabledBoundsSafetyChecks(Result, Diags,
                                       IndividualChecksExplicitlyDisabled);
  return Result;
}

} // namespace driver

} // namespace clang
