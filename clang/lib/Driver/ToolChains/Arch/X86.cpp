//===--- X86.cpp - X86 Helpers for Tools ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "ToolChains/CommonArgs.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/ArgList.h"

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

const char *x86::getX86TargetCPU(const ArgList &Args,
                                 const llvm::Triple &Triple) {
  if (const Arg *A = Args.getLastArg(clang::driver::options::OPT_march_EQ)) {
    if (StringRef(A->getValue()) != "native") {
      if (Triple.isOSDarwin() && Triple.getArchName() == "x86_64h")
        return "core-avx2";

      return A->getValue();
    }

    // FIXME: Reject attempts to use -march=native unless the target matches
    // the host.
    //
    // FIXME: We should also incorporate the detected target features for use
    // with -native.
    std::string CPU = llvm::sys::getHostCPUName();
    if (!CPU.empty() && CPU != "generic")
      return Args.MakeArgString(CPU);
  }

  if (const Arg *A = Args.getLastArg(options::OPT__SLASH_arch)) {
    // Mapping built by referring to X86TargetInfo::getDefaultFeatures().
    StringRef Arch = A->getValue();
    const char *CPU;
    if (Triple.getArch() == llvm::Triple::x86) {
      CPU = llvm::StringSwitch<const char *>(Arch)
                .Case("IA32", "i386")
                .Case("SSE", "pentium3")
                .Case("SSE2", "pentium4")
                .Case("AVX", "sandybridge")
                .Case("AVX2", "haswell")
                .Default(nullptr);
    } else {
      CPU = llvm::StringSwitch<const char *>(Arch)
                .Case("AVX", "sandybridge")
                .Case("AVX2", "haswell")
                .Default(nullptr);
    }
    if (CPU)
      return CPU;
  }

  // Select the default CPU if none was given (or detection failed).

  if (Triple.getArch() != llvm::Triple::x86_64 &&
      Triple.getArch() != llvm::Triple::x86)
    return nullptr; // This routine is only handling x86 targets.

  bool Is64Bit = Triple.getArch() == llvm::Triple::x86_64;

  // FIXME: Need target hooks.
  if (Triple.isOSDarwin()) {
    if (Triple.getArchName() == "x86_64h")
      return "core-avx2";
    // macosx10.12 drops support for all pre-Penryn Macs.
    // Simulators can still run on 10.11 though, like Xcode.
    if (Triple.isMacOSX() && !Triple.isOSVersionLT(10, 12))
      return "penryn";
    // The oldest x86_64 Macs have core2/Merom; the oldest x86 Macs have Yonah.
    return Is64Bit ? "core2" : "yonah";
  }

  // Set up default CPU name for PS4 compilers.
  if (Triple.isPS4CPU())
    return "btver2";

  // On Android use targets compatible with gcc
  if (Triple.isAndroid())
    return Is64Bit ? "x86-64" : "i686";

  // Everything else goes to x86-64 in 64-bit mode.
  if (Is64Bit)
    return "x86-64";

  switch (Triple.getOS()) {
  case llvm::Triple::FreeBSD:
  case llvm::Triple::NetBSD:
  case llvm::Triple::OpenBSD:
    return "i486";
  case llvm::Triple::Haiku:
    return "i586";
  case llvm::Triple::Bitrig:
    return "i686";
  default:
    // Fallback to p4.
    return "pentium4";
  }
}

void x86::getX86TargetFeatures(const Driver &D, const llvm::Triple &Triple,
                               const ArgList &Args,
                               std::vector<StringRef> &Features) {
  // If -march=native, autodetect the feature list.
  if (const Arg *A = Args.getLastArg(clang::driver::options::OPT_march_EQ)) {
    if (StringRef(A->getValue()) == "native") {
      llvm::StringMap<bool> HostFeatures;
      if (llvm::sys::getHostCPUFeatures(HostFeatures))
        for (auto &F : HostFeatures)
          Features.push_back(
              Args.MakeArgString((F.second ? "+" : "-") + F.first()));
    }
  }

  if (Triple.getArchName() == "x86_64h") {
    // x86_64h implies quite a few of the more modern subtarget features
    // for Haswell class CPUs, but not all of them. Opt-out of a few.
    Features.push_back("-rdrnd");
    Features.push_back("-aes");
    Features.push_back("-pclmul");
    Features.push_back("-rtm");
    Features.push_back("-hle");
    Features.push_back("-fsgsbase");
  }

  const llvm::Triple::ArchType ArchType = Triple.getArch();
  // Add features to be compatible with gcc for Android.
  if (Triple.isAndroid()) {
    if (ArchType == llvm::Triple::x86_64) {
      Features.push_back("+sse4.2");
      Features.push_back("+popcnt");
    } else
      Features.push_back("+ssse3");
  }

  // Set features according to the -arch flag on MSVC.
  if (Arg *A = Args.getLastArg(options::OPT__SLASH_arch)) {
    StringRef Arch = A->getValue();
    bool ArchUsed = false;
    // First, look for flags that are shared in x86 and x86-64.
    if (ArchType == llvm::Triple::x86_64 || ArchType == llvm::Triple::x86) {
      if (Arch == "AVX" || Arch == "AVX2") {
        ArchUsed = true;
        Features.push_back(Args.MakeArgString("+" + Arch.lower()));
      }
    }
    // Then, look for x86-specific flags.
    if (ArchType == llvm::Triple::x86) {
      if (Arch == "IA32") {
        ArchUsed = true;
      } else if (Arch == "SSE" || Arch == "SSE2") {
        ArchUsed = true;
        Features.push_back(Args.MakeArgString("+" + Arch.lower()));
      }
    }
    if (!ArchUsed)
      D.Diag(clang::diag::warn_drv_unused_argument) << A->getAsString(Args);
  }

  // Now add any that the user explicitly requested on the command line,
  // which may override the defaults.
  handleTargetFeaturesGroup(Args, Features, options::OPT_m_x86_Features_Group);
}
