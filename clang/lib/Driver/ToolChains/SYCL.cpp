//===--- SYCL.cpp - SYCL Tool and ToolChain Implementations -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "SYCL.h"
#include "clang/Driver/CommonArgs.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

SYCLInstallationDetector::SYCLInstallationDetector(
    const Driver &D, const llvm::Triple &HostTriple,
    const llvm::opt::ArgList &Args)
    : D(D) {
  // When -fsycl is active, locate the SYCL runtime library and record its
  // directory in SYCLRTLibPath for use by the linker.
  StringRef SysRoot = D.SysRoot;
  SmallString<128> DriverDir(D.Dir);

  if (HostTriple.isWindowsMSVCEnvironment() ||
      HostTriple.isWindowsItaniumEnvironment()) {
    // Windows: Check for LLVMSYCL.lib
    // NOTE: Only checks for LLVMSYCL.lib existence (release variant).
    // Debug vs release library selection happens at link time based on CRT
    // flags.
    if (DriverDir.starts_with(SysRoot) &&
        Args.hasFlag(options::OPT_fsycl, options::OPT_fno_sycl, false)) {
      SmallString<128> LibDir(DriverDir);
      llvm::sys::path::append(LibDir, "..", "lib");

      // Verify SYCL runtime library exists
      SmallString<128> SYCLLibPath(LibDir);
      llvm::sys::path::append(SYCLLibPath, "LLVMSYCL.lib");

      if (D.getVFS().exists(SYCLLibPath))
        SYCLRTLibPath = LibDir;
    }
  } else {
    // Linux/Unix: Check for libLLVMSYCL.so
    SmallString<128> LibPath(DriverDir);
    llvm::sys::path::append(LibPath, "..", "lib", HostTriple.str(),
                            "libLLVMSYCL.so");
    // Flat lib path for LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF builds,
    // where the library is installed directly in lib/ with no triple subdir.
    SmallString<128> FlatLibPath(DriverDir);
    llvm::sys::path::append(FlatLibPath, "..", "lib", "libLLVMSYCL.so");

    if (DriverDir.starts_with(SysRoot) &&
        Args.hasFlag(options::OPT_fsycl, options::OPT_fno_sycl, false)) {
      // LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON: library is in lib/<triple>/
      if (D.getVFS().exists(LibPath))
        llvm::sys::path::append(DriverDir, "..", "lib", HostTriple.str());
      // LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF: library is in lib/
      else if (D.getVFS().exists(FlatLibPath))
        llvm::sys::path::append(DriverDir, "..", "lib");
      else
        return; // Neither path exists : broken install, leave SYCLRTLibPath
                // unset

      SYCLRTLibPath = DriverDir;
    }
  }
}

void SYCLInstallationDetector::addSYCLIncludeArgs(
    const ArgList &DriverArgs, ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nobuiltininc))
    return;

  // Add the SYCL header search locations.
  // These are included for both SYCL host and device compilations.
  SmallString<128> IncludePath(D.Dir);
  llvm::sys::path::append(IncludePath, "..", "include");
  CC1Args.push_back("-internal-isystem");
  CC1Args.push_back(DriverArgs.MakeArgString(IncludePath));
}

// Unsupported options for SYCL device compilation.
static ArrayRef<options::ID> getUnsupportedOpts() {
  static constexpr options::ID UnsupportedOpts[] = {
      options::OPT_fsanitize_EQ,      // -fsanitize
      options::OPT_fcf_protection_EQ, // -fcf-protection
      options::OPT_fprofile_generate,
      options::OPT_fprofile_generate_EQ,
      options::OPT_fno_profile_generate, // -f[no-]profile-generate
      options::OPT_ftest_coverage,
      options::OPT_fno_test_coverage, // -f[no-]test-coverage
      options::OPT_fcoverage_mapping,
      options::OPT_fno_coverage_mapping, // -f[no-]coverage-mapping
      options::OPT_coverage,             // --coverage
      options::OPT_fprofile_instr_generate,
      options::OPT_fprofile_instr_generate_EQ,
      options::OPT_fno_profile_instr_generate, // -f[no-]profile-instr-generate
      options::OPT_fprofile_arcs,
      options::OPT_fno_profile_arcs, // -f[no-]profile-arcs
      options::OPT_fcreate_profile,  // -fcreate-profile
      options::OPT_fprofile_instr_use,
      options::OPT_fprofile_instr_use_EQ, // -fprofile-instr-use
      options::OPT_fcs_profile_generate,  // -fcs-profile-generate
      options::OPT_fcs_profile_generate_EQ,
  };
  return UnsupportedOpts;
}

SYCLToolChain::SYCLToolChain(const Driver &D, const llvm::Triple &Triple,
                             const ToolChain &HostTC, const ArgList &Args)
    : ToolChain(D, Triple, Args), HostTC(HostTC),
      SYCLInstallation(D, Triple, Args) {
  // Lookup binaries into the driver directory, this is used to discover any
  // dependent SYCL offload compilation tools.
  getProgramPaths().push_back(getDriver().Dir);

  // Diagnose unsupported options only once.
  for (OptSpecifier Opt : getUnsupportedOpts()) {
    if (const Arg *A = Args.getLastArg(Opt)) {
      D.Diag(clang::diag::warn_drv_unsupported_option_for_target)
          << A->getAsString(Args) << getTriple().str();
    }
  }
}

void SYCLToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs, llvm::opt::ArgStringList &CC1Args,
    Action::OffloadKind DeviceOffloadingKind) const {
  HostTC.addClangTargetOptions(DriverArgs, CC1Args, DeviceOffloadingKind);
}

llvm::opt::DerivedArgList *
SYCLToolChain::TranslateArgs(const llvm::opt::DerivedArgList &Args,
                             StringRef BoundArch,
                             Action::OffloadKind DeviceOffloadKind) const {
  DerivedArgList *DAL =
      HostTC.TranslateArgs(Args, BoundArch, DeviceOffloadKind);

  bool IsNewDAL = false;
  if (!DAL) {
    DAL = new DerivedArgList(Args.getBaseArgs());
    IsNewDAL = true;
  }

  for (Arg *A : Args) {
    // Filter out any options we do not want to pass along to the device
    // compilation.
    auto Opt(A->getOption());
    bool Unsupported = false;
    for (OptSpecifier UnsupportedOpt : getUnsupportedOpts()) {
      if (Opt.matches(UnsupportedOpt)) {
        if (Opt.getID() == options::OPT_fsanitize_EQ &&
            A->getValues().size() == 1) {
          std::string SanitizeVal = A->getValue();
          if (SanitizeVal == "address") {
            if (IsNewDAL)
              DAL->append(A);
            continue;
          }
        }
        if (!IsNewDAL)
          DAL->eraseArg(Opt.getID());
        Unsupported = true;
      }
    }
    if (Unsupported)
      continue;
    if (IsNewDAL)
      DAL->append(A);
  }

  const OptTable &Opts = getDriver().getOpts();
  if (!BoundArch.empty()) {
    DAL->eraseArg(options::OPT_march_EQ);
    DAL->AddJoinedArg(nullptr, Opts.getOption(options::OPT_march_EQ),
                      BoundArch);
  }
  return DAL;
}

void SYCLToolChain::addClangWarningOptions(ArgStringList &CC1Args) const {
  HostTC.addClangWarningOptions(CC1Args);
}

ToolChain::CXXStdlibType
SYCLToolChain::GetCXXStdlibType(const ArgList &Args) const {
  return HostTC.GetCXXStdlibType(Args);
}

void SYCLToolChain::addSYCLIncludeArgs(const ArgList &DriverArgs,
                                       ArgStringList &CC1Args) const {
  SYCLInstallation.addSYCLIncludeArgs(DriverArgs, CC1Args);
}

void SYCLToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                              ArgStringList &CC1Args) const {
  HostTC.AddClangSystemIncludeArgs(DriverArgs, CC1Args);
}

void SYCLToolChain::AddClangCXXStdlibIncludeArgs(const ArgList &Args,
                                                 ArgStringList &CC1Args) const {
  HostTC.AddClangCXXStdlibIncludeArgs(Args, CC1Args);
}
