//===--- M68k.cpp - M68k Helpers for Tools -------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "M68k.h"
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

/// getM68kTargetCPU - Get the (LLVM) name of the 68000 cpu we are targeting.
std::string m68k::getM68kTargetCPU(const ArgList &Args) {
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
        .Cases("m68000", "68000", "M68000")
        .Cases("m68010", "68010", "M68010")
        .Cases("m68020", "68020", "M68020")
        .Cases("m68030", "68030", "M68030")
        .Cases("m68040", "68040", "M68040")
        .Cases("m68060", "68060", "M68060")
        .Default(CPUName.str());
  }
  // FIXME: Throw error when multiple sub-architecture flag exist
  if (Args.hasArg(clang::driver::options::OPT_m68000))
    return "M68000";
  if (Args.hasArg(clang::driver::options::OPT_m68010))
    return "M68010";
  if (Args.hasArg(clang::driver::options::OPT_m68020))
    return "M68020";
  if (Args.hasArg(clang::driver::options::OPT_m68030))
    return "M68030";
  if (Args.hasArg(clang::driver::options::OPT_m68040))
    return "M68040";
  if (Args.hasArg(clang::driver::options::OPT_m68060))
    return "M68060";

  return "";
}

void m68k::getM68kTargetFeatures(const Driver &D, const llvm::Triple &Triple,
                                 const ArgList &Args,
                                 std::vector<StringRef> &Features) {

  m68k::FloatABI FloatABI = m68k::getM68kFloatABI(D, Args);
  if (FloatABI == m68k::FloatABI::Soft)
    Features.push_back("-hard-float");
}

m68k::FloatABI m68k::getM68kFloatABI(const Driver &D, const ArgList &Args) {
  m68k::FloatABI ABI = m68k::FloatABI::Invalid;
  if (Arg *A =
          Args.getLastArg(options::OPT_msoft_float, options::OPT_mhard_float)) {

    if (A->getOption().matches(options::OPT_msoft_float))
      ABI = m68k::FloatABI::Soft;
    else if (A->getOption().matches(options::OPT_mhard_float))
      ABI = m68k::FloatABI::Hard;
  }

  // If unspecified, choose the default based on the platform.
  if (ABI == m68k::FloatABI::Invalid)
    ABI = m68k::FloatABI::Hard;

  return ABI;
}
