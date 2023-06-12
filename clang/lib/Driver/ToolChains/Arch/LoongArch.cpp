//===--- LoongArch.cpp - LoongArch Helpers for Tools ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LoongArch.h"
#include "ToolChains/CommonArgs.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/TargetParser/LoongArchTargetParser.h"

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

StringRef loongarch::getLoongArchABI(const Driver &D, const ArgList &Args,
                                     const llvm::Triple &Triple) {
  assert((Triple.getArch() == llvm::Triple::loongarch32 ||
          Triple.getArch() == llvm::Triple::loongarch64) &&
         "Unexpected triple");
  bool IsLA32 = Triple.getArch() == llvm::Triple::loongarch32;

  // Check -m*-float firstly since they have highest priority.
  if (const Arg *A = Args.getLastArg(options::OPT_mdouble_float,
                                     options::OPT_msingle_float,
                                     options::OPT_msoft_float)) {
    if (A->getOption().matches(options::OPT_mdouble_float))
      return IsLA32 ? "ilp32d" : "lp64d";
    if (A->getOption().matches(options::OPT_msingle_float))
      return IsLA32 ? "ilp32f" : "lp64f";
    if (A->getOption().matches(options::OPT_msoft_float))
      return IsLA32 ? "ilp32s" : "lp64s";
  }

  // If `-mabi=` is specified, use it.
  if (const Arg *A = Args.getLastArg(options::OPT_mabi_EQ))
    return A->getValue();

  // Select abi based on -mfpu=xx.
  if (const Arg *A = Args.getLastArg(options::OPT_mfpu_EQ)) {
    StringRef FPU = A->getValue();
    if (FPU == "64")
      return IsLA32 ? "ilp32d" : "lp64d";
    if (FPU == "32")
      return IsLA32 ? "ilp32f" : "lp64f";
    if (FPU == "0" || FPU == "none")
      return IsLA32 ? "ilp32s" : "lp64s";
    D.Diag(diag::err_drv_loongarch_invalid_mfpu_EQ) << FPU;
  }

  // Choose a default based on the triple.
  // Honor the explicit ABI modifier suffix in triple's environment part if
  // present, falling back to {ILP32,LP64}D otherwise.
  switch (Triple.getEnvironment()) {
  case llvm::Triple::GNUSF:
    return IsLA32 ? "ilp32s" : "lp64s";
  case llvm::Triple::GNUF32:
    return IsLA32 ? "ilp32f" : "lp64f";
  case llvm::Triple::GNUF64:
    // This was originally permitted (and indeed the canonical way) to
    // represent the {ILP32,LP64}D ABIs, but in Feb 2023 Loongson decided to
    // drop the explicit suffix in favor of unmarked `-gnu` for the
    // "general-purpose" ABIs, among other non-technical reasons.
    //
    // The spec change did not mention whether existing usages of "gnuf64"
    // shall remain valid or not, so we are going to continue recognizing it
    // for some time, until it is clear that everyone else has migrated away
    // from it.
    [[fallthrough]];
  case llvm::Triple::GNU:
  default:
    return IsLA32 ? "ilp32d" : "lp64d";
  }
}

void loongarch::getLoongArchTargetFeatures(const Driver &D,
                                           const llvm::Triple &Triple,
                                           const ArgList &Args,
                                           std::vector<StringRef> &Features) {
  StringRef ArchName;
  llvm::LoongArch::ArchKind ArchKind = llvm::LoongArch::ArchKind::AK_INVALID;
  if (const Arg *A = Args.getLastArg(options::OPT_march_EQ)) {
    ArchKind = llvm::LoongArch::parseArch(A->getValue());
    if (ArchKind == llvm::LoongArch::ArchKind::AK_INVALID) {
      D.Diag(clang::diag::err_drv_invalid_arch_name) << A->getAsString(Args);
      return;
    }
    ArchName = A->getValue();
  }

  // TODO: handle -march=native and -mtune=xx.

  // Select a default arch name.
  if (ArchName.empty() && Triple.getArch() == llvm::Triple::loongarch64)
    ArchName = "loongarch64";

  if (!ArchName.empty())
    llvm::LoongArch::getArchFeatures(ArchName, Features);

  // Select floating-point features determined by -mdouble-float,
  // -msingle-float, -msoft-float and -mfpu.
  // Note: -m*-float wins any other options.
  if (const Arg *A = Args.getLastArg(options::OPT_mdouble_float,
                                     options::OPT_msingle_float,
                                     options::OPT_msoft_float)) {
    if (A->getOption().matches(options::OPT_mdouble_float)) {
      Features.push_back("+f");
      Features.push_back("+d");
    } else if (A->getOption().matches(options::OPT_msingle_float)) {
      Features.push_back("+f");
      Features.push_back("-d");
    } else /*Soft-float*/ {
      Features.push_back("-f");
      Features.push_back("-d");
    }
  } else if (const Arg *A = Args.getLastArg(options::OPT_mfpu_EQ)) {
    StringRef FPU = A->getValue();
    if (FPU == "64") {
      Features.push_back("+f");
      Features.push_back("+d");
    } else if (FPU == "32") {
      Features.push_back("+f");
      Features.push_back("-d");
    } else if (FPU == "0" || FPU == "none") {
      Features.push_back("-f");
      Features.push_back("-d");
    } else {
      D.Diag(diag::err_drv_loongarch_invalid_mfpu_EQ) << FPU;
    }
  }

  // Select the `ual` feature determined by -m[no-]unaligned-access
  // or the alias -m[no-]strict-align.
  AddTargetFeature(Args, Features, options::OPT_munaligned_access,
                   options::OPT_mno_unaligned_access, "ual");
}
