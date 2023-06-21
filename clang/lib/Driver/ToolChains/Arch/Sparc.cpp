//===--- Sparc.cpp - Tools Implementations ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Sparc.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/ArgList.h"
#include "llvm/TargetParser/Host.h"

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

const char *sparc::getSparcAsmModeForCPU(StringRef Name,
                                         const llvm::Triple &Triple) {
  if (Triple.getArch() == llvm::Triple::sparcv9) {
    const char *DefV9CPU;

    if (Triple.isOSLinux() || Triple.isOSFreeBSD() || Triple.isOSOpenBSD())
      DefV9CPU = "-Av9a";
    else
      DefV9CPU = "-Av9";

    return llvm::StringSwitch<const char *>(Name)
        .Case("niagara", "-Av9b")
        .Case("niagara2", "-Av9b")
        .Case("niagara3", "-Av9d")
        .Case("niagara4", "-Av9d")
        .Default(DefV9CPU);
  } else {
    return llvm::StringSwitch<const char *>(Name)
        .Case("v8", "-Av8")
        .Case("supersparc", "-Av8")
        .Case("sparclite", "-Asparclite")
        .Case("f934", "-Asparclite")
        .Case("hypersparc", "-Av8")
        .Case("sparclite86x", "-Asparclite")
        .Case("sparclet", "-Asparclet")
        .Case("tsc701", "-Asparclet")
        .Case("v9", "-Av8plus")
        .Case("ultrasparc", "-Av8plus")
        .Case("ultrasparc3", "-Av8plus")
        .Case("niagara", "-Av8plusb")
        .Case("niagara2", "-Av8plusb")
        .Case("niagara3", "-Av8plusd")
        .Case("niagara4", "-Av8plusd")
        .Case("ma2100", "-Aleon")
        .Case("ma2150", "-Aleon")
        .Case("ma2155", "-Aleon")
        .Case("ma2450", "-Aleon")
        .Case("ma2455", "-Aleon")
        .Case("ma2x5x", "-Aleon")
        .Case("ma2080", "-Aleon")
        .Case("ma2085", "-Aleon")
        .Case("ma2480", "-Aleon")
        .Case("ma2485", "-Aleon")
        .Case("ma2x8x", "-Aleon")
        .Case("myriad2", "-Aleon")
        .Case("myriad2.1", "-Aleon")
        .Case("myriad2.2", "-Aleon")
        .Case("myriad2.3", "-Aleon")
        .Case("leon2", "-Av8")
        .Case("at697e", "-Av8")
        .Case("at697f", "-Av8")
        .Case("leon3", "-Aleon")
        .Case("ut699", "-Av8")
        .Case("gr712rc", "-Aleon")
        .Case("leon4", "-Aleon")
        .Case("gr740", "-Aleon")
        .Default("-Av8");
  }
}

sparc::FloatABI sparc::getSparcFloatABI(const Driver &D,
                                        const ArgList &Args) {
  sparc::FloatABI ABI = sparc::FloatABI::Invalid;
  if (Arg *A = Args.getLastArg(options::OPT_msoft_float, options::OPT_mno_fpu,
                               options::OPT_mhard_float, options::OPT_mfpu,
                               options::OPT_mfloat_abi_EQ)) {
    if (A->getOption().matches(options::OPT_msoft_float) ||
        A->getOption().matches(options::OPT_mno_fpu))
      ABI = sparc::FloatABI::Soft;
    else if (A->getOption().matches(options::OPT_mhard_float) ||
             A->getOption().matches(options::OPT_mfpu))
      ABI = sparc::FloatABI::Hard;
    else {
      ABI = llvm::StringSwitch<sparc::FloatABI>(A->getValue())
                .Case("soft", sparc::FloatABI::Soft)
                .Case("hard", sparc::FloatABI::Hard)
                .Default(sparc::FloatABI::Invalid);
      if (ABI == sparc::FloatABI::Invalid &&
          !StringRef(A->getValue()).empty()) {
        D.Diag(clang::diag::err_drv_invalid_mfloat_abi) << A->getAsString(Args);
        ABI = sparc::FloatABI::Hard;
      }
    }
  }

  // If unspecified, choose the default based on the platform.
  // Only the hard-float ABI on Sparc is standardized, and it is the
  // default. GCC also supports a nonstandard soft-float ABI mode, also
  // implemented in LLVM. However as this is not standard we set the default
  // to be hard-float.
  if (ABI == sparc::FloatABI::Invalid) {
    ABI = sparc::FloatABI::Hard;
  }

  return ABI;
}

std::string sparc::getSparcTargetCPU(const Driver &D, const ArgList &Args,
                                     const llvm::Triple &Triple) {
  if (const Arg *A = Args.getLastArg(clang::driver::options::OPT_mcpu_EQ)) {
    StringRef CPUName = A->getValue();
    if (CPUName == "native") {
      std::string CPU = std::string(llvm::sys::getHostCPUName());
      if (!CPU.empty() && CPU != "generic")
        return CPU;
      return "";
    }
    return std::string(CPUName);
  }

  if (Triple.getArch() == llvm::Triple::sparc && Triple.isOSSolaris())
    return "v9";
  return "";
}

void sparc::getSparcTargetFeatures(const Driver &D, const ArgList &Args,
                                   std::vector<StringRef> &Features) {
  sparc::FloatABI FloatABI = sparc::getSparcFloatABI(D, Args);
  if (FloatABI == sparc::FloatABI::Soft)
    Features.push_back("+soft-float");

  if (Arg *A = Args.getLastArg(options::OPT_mfsmuld, options::OPT_mno_fsmuld)) {
    if (A->getOption().matches(options::OPT_mfsmuld))
      Features.push_back("+fsmuld");
    else
      Features.push_back("-fsmuld");
  }

  if (Arg *A = Args.getLastArg(options::OPT_mpopc, options::OPT_mno_popc)) {
    if (A->getOption().matches(options::OPT_mpopc))
      Features.push_back("+popc");
    else
      Features.push_back("-popc");
  }

  if (Arg *A = Args.getLastArg(options::OPT_mvis, options::OPT_mno_vis)) {
    if (A->getOption().matches(options::OPT_mvis))
      Features.push_back("+vis");
    else
      Features.push_back("-vis");
  }

  if (Arg *A = Args.getLastArg(options::OPT_mvis2, options::OPT_mno_vis2)) {
    if (A->getOption().matches(options::OPT_mvis2))
      Features.push_back("+vis2");
    else
      Features.push_back("-vis2");
  }

  if (Arg *A = Args.getLastArg(options::OPT_mvis3, options::OPT_mno_vis3)) {
    if (A->getOption().matches(options::OPT_mvis3))
      Features.push_back("+vis3");
    else
      Features.push_back("-vis3");
  }

  if (Arg *A = Args.getLastArg(options::OPT_mhard_quad_float,
                               options::OPT_msoft_quad_float)) {
    if (A->getOption().matches(options::OPT_mhard_quad_float))
      Features.push_back("+hard-quad-float");
    else
      Features.push_back("-hard-quad-float");
  }
}
