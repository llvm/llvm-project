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

/// getM88KTargetCPU - Get the (LLVM) name of the 88000 cpu we are targeting.
std::string m88k::getM88kTargetCPU(const ArgList &Args) {
  if (Arg *A = Args.getLastArg(clang::driver::options::OPT_mcpu_EQ)) {
    // The canonical CPU name is captalize. However, we allow
    // starting with lower case or numbers only
    StringRef CPUName = A->getValue();

    if (CPUName == "native") {
      std::string CPU = std::string(llvm::sys::getHostCPUName());
      if (!CPU.empty() && CPU != "generic")
        return CPU;
    }

    if (CPUName == "common")
      return "generic";

    return llvm::StringSwitch<std::string>(CPUName)
        .Cases("m88000", "88000", "M88000")
        .Cases("m88100", "88100", "M88100")
        .Cases("m88110", "88110", "M88110")
        .Default(CPUName.str());
  }
  // FIXME: Throw error when multiple sub-architecture flag exist
  if (Args.hasArg(clang::driver::options::OPT_m88000))
    return "M88000";
  if (Args.hasArg(clang::driver::options::OPT_m88100))
    return "M88100";
  if (Args.hasArg(clang::driver::options::OPT_m88110))
    return "M88110";

  return "";
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
