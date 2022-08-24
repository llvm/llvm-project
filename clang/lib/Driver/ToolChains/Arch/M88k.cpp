//===--- M88k.cpp - M88k Helpers for Tools --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "M88k.h"
#include "ToolChains/CommonArgs.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Regex.h"
#include <sstream>

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

static StringRef normalizeCPU(StringRef CPUName) {
  if (CPUName == "native") {
    StringRef CPU = std::string(llvm::sys::getHostCPUName());
    if (!CPU.empty() && CPU != "generic")
      return CPU;
  }

  return llvm::StringSwitch<StringRef>(CPUName)
      .Cases("mc88000", "m88000", "88000", "generic", "mc88000")
      .Cases("mc88100", "m88100", "88100", "mc88100")
      .Cases("mc88110", "m88110", "88110", "mc88110")
      .Default(CPUName);
}

/// getM88KTargetCPU - Get the (LLVM) name of the 88000 cpu we are targeting.
StringRef m88k::getM88kTargetCPU(const ArgList &Args) {
  Arg *A = Args.getLastArg(options::OPT_m88000, options::OPT_m88100,
                           options::OPT_m88110, options::OPT_mcpu_EQ);
  if (!A)
    return StringRef();

  switch (A->getOption().getID()) {
  case options::OPT_m88000:
    return "mc88000";
  case options::OPT_m88100:
    return "mc88100";
  case options::OPT_m88110:
    return "mc88110";
  case options::OPT_mcpu_EQ:
    return normalizeCPU(A->getValue());
  default:
    llvm_unreachable("Impossible option ID");
  }
}

StringRef m88k::getM88kTuneCPU(const ArgList &Args) {
  if (const Arg *A = Args.getLastArg(options::OPT_mtune_EQ))
    return normalizeCPU(A->getValue());
  return StringRef();
}

void m88k::getM88kTargetFeatures(const Driver &D, const llvm::Triple &Triple,
                                 const ArgList &Args,
                                 std::vector<StringRef> &Features) {
  m88k::FloatABI FloatABI = m88k::getM88kFloatABI(D, Args);
  if (FloatABI == m88k::FloatABI::Soft)
    Features.push_back("-hard-float");
}

m88k::FloatABI m88k::getM88kFloatABI(const Driver &D, const ArgList &Args) {
  m88k::FloatABI ABI = m88k::FloatABI::Invalid;
  if (Arg *A =
          Args.getLastArg(options::OPT_msoft_float, options::OPT_mhard_float)) {

    if (A->getOption().matches(options::OPT_msoft_float))
      ABI = m88k::FloatABI::Soft;
    else if (A->getOption().matches(options::OPT_mhard_float))
      ABI = m88k::FloatABI::Hard;
  }

  // If unspecified, choose the default based on the platform.
  if (ABI == m88k::FloatABI::Invalid)
    ABI = m88k::FloatABI::Hard;

  return ABI;
}
