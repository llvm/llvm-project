//===--- AArch64.cpp - AArch64 (not ARM) Helpers for Tools ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/TargetParser.h"

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

/// getAArch64TargetCPU - Get the (LLVM) name of the AArch64 cpu we are
/// targeting. Set \p A to the Arg corresponding to the -mcpu or -mtune
/// arguments if they are provided, or to nullptr otherwise.
std::string aarch64::getAArch64TargetCPU(const ArgList &Args, Arg *&A) {
  std::string CPU;
  // If we have -mtune or -mcpu, use that.
  if ((A = Args.getLastArg(clang::driver::options::OPT_mtune_EQ))) {
    CPU = StringRef(A->getValue()).lower();
  } else if ((A = Args.getLastArg(options::OPT_mcpu_EQ))) {
    StringRef Mcpu = A->getValue();
    CPU = Mcpu.split("+").first.lower();
  }

  // Handle CPU name is 'native'.
  if (CPU == "native")
    return llvm::sys::getHostCPUName();
  else if (CPU.size())
    return CPU;

  // Make sure we pick "cyclone" if -arch is used.
  // FIXME: Should this be picked by checking the target triple instead?
  if (Args.getLastArg(options::OPT_arch))
    return "cyclone";

  return "generic";
}

// Decode AArch64 features from string like +[no]featureA+[no]featureB+...
static bool DecodeAArch64Features(const Driver &D, StringRef text,
                                  std::vector<StringRef> &Features) {
  SmallVector<StringRef, 8> Split;
  text.split(Split, StringRef("+"), -1, false);

  for (StringRef Feature : Split) {
    StringRef FeatureName = llvm::AArch64::getArchExtFeature(Feature);
    if (!FeatureName.empty())
      Features.push_back(FeatureName);
    else if (Feature == "neon" || Feature == "noneon")
      D.Diag(clang::diag::err_drv_no_neon_modifier);
    else
      return false;
  }
  return true;
}

// Check if the CPU name and feature modifiers in -mcpu are legal. If yes,
// decode CPU and feature.
static bool DecodeAArch64Mcpu(const Driver &D, StringRef Mcpu, StringRef &CPU,
                              std::vector<StringRef> &Features) {
  std::pair<StringRef, StringRef> Split = Mcpu.split("+");
  CPU = Split.first;

  if (CPU == "generic") {
    Features.push_back("+neon");
  } else {
    unsigned ArchKind = llvm::AArch64::parseCPUArch(CPU);
    if (!llvm::AArch64::getArchFeatures(ArchKind, Features))
      return false;

    unsigned Extension = llvm::AArch64::getDefaultExtensions(CPU, ArchKind);
    if (!llvm::AArch64::getExtensionFeatures(Extension, Features))
      return false;
   }

  if (Split.second.size() && !DecodeAArch64Features(D, Split.second, Features))
    return false;

  return true;
}

static bool
getAArch64ArchFeaturesFromMarch(const Driver &D, StringRef March,
                                const ArgList &Args,
                                std::vector<StringRef> &Features) {
  std::string MarchLowerCase = March.lower();
  std::pair<StringRef, StringRef> Split = StringRef(MarchLowerCase).split("+");

  unsigned ArchKind = llvm::AArch64::parseArch(Split.first);
  if (ArchKind == static_cast<unsigned>(llvm::AArch64::ArchKind::AK_INVALID) ||
      !llvm::AArch64::getArchFeatures(ArchKind, Features) ||
      (Split.second.size() && !DecodeAArch64Features(D, Split.second, Features)))
    return false;

  return true;
}

static bool
getAArch64ArchFeaturesFromMcpu(const Driver &D, StringRef Mcpu,
                               const ArgList &Args,
                               std::vector<StringRef> &Features) {
  StringRef CPU;
  std::string McpuLowerCase = Mcpu.lower();
  if (!DecodeAArch64Mcpu(D, McpuLowerCase, CPU, Features))
    return false;

  return true;
}

static bool
getAArch64MicroArchFeaturesFromMtune(const Driver &D, StringRef Mtune,
                                     const ArgList &Args,
                                     std::vector<StringRef> &Features) {
  std::string MtuneLowerCase = Mtune.lower();
  // Handle CPU name is 'native'.
  if (MtuneLowerCase == "native")
    MtuneLowerCase = llvm::sys::getHostCPUName();
  if (MtuneLowerCase == "cyclone") {
    Features.push_back("+zcm");
    Features.push_back("+zcz");
  }
  return true;
}

static bool
getAArch64MicroArchFeaturesFromMcpu(const Driver &D, StringRef Mcpu,
                                    const ArgList &Args,
                                    std::vector<StringRef> &Features) {
  StringRef CPU;
  std::vector<StringRef> DecodedFeature;
  std::string McpuLowerCase = Mcpu.lower();
  if (!DecodeAArch64Mcpu(D, McpuLowerCase, CPU, DecodedFeature))
    return false;

  return getAArch64MicroArchFeaturesFromMtune(D, CPU, Args, Features);
}

void aarch64::getAArch64TargetFeatures(const Driver &D, const ArgList &Args,
                                       std::vector<StringRef> &Features) {
  Arg *A;
  bool success = true;
  // Enable NEON by default.
  Features.push_back("+neon");
  if ((A = Args.getLastArg(options::OPT_march_EQ)))
    success = getAArch64ArchFeaturesFromMarch(D, A->getValue(), Args, Features);
  else if ((A = Args.getLastArg(options::OPT_mcpu_EQ)))
    success = getAArch64ArchFeaturesFromMcpu(D, A->getValue(), Args, Features);
  else if (Args.hasArg(options::OPT_arch))
    success = getAArch64ArchFeaturesFromMcpu(D, getAArch64TargetCPU(Args, A),
                                             Args, Features);

  if (success && (A = Args.getLastArg(clang::driver::options::OPT_mtune_EQ)))
    success =
        getAArch64MicroArchFeaturesFromMtune(D, A->getValue(), Args, Features);
  else if (success && (A = Args.getLastArg(options::OPT_mcpu_EQ)))
    success =
        getAArch64MicroArchFeaturesFromMcpu(D, A->getValue(), Args, Features);
  else if (success && Args.hasArg(options::OPT_arch))
    success = getAArch64MicroArchFeaturesFromMcpu(
        D, getAArch64TargetCPU(Args, A), Args, Features);

  if (!success)
    D.Diag(diag::err_drv_clang_unsupported) << A->getAsString(Args);

  if (Args.getLastArg(options::OPT_mgeneral_regs_only)) {
    Features.push_back("-fp-armv8");
    Features.push_back("-crypto");
    Features.push_back("-neon");
  }

  // En/disable crc
  if (Arg *A = Args.getLastArg(options::OPT_mcrc, options::OPT_mnocrc)) {
    if (A->getOption().matches(options::OPT_mcrc))
      Features.push_back("+crc");
    else
      Features.push_back("-crc");
  }

  if (Arg *A = Args.getLastArg(options::OPT_mno_unaligned_access,
                               options::OPT_munaligned_access))
    if (A->getOption().matches(options::OPT_mno_unaligned_access))
      Features.push_back("+strict-align");

  if (Args.hasArg(options::OPT_ffixed_x18))
    Features.push_back("+reserve-x18");

  if (Args.hasArg(options::OPT_mno_neg_immediates))
    Features.push_back("+no-neg-immediates");
}
