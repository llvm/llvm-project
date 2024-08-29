//===--- SYCL.cpp - SYCL Tool and ToolChain Implementations -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "SYCL.h"
#include "CommonArgs.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/InputInfo.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <sstream>

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

SYCLInstallationDetector::SYCLInstallationDetector(const Driver &D)
    : D(D), InstallationCandidates() {
  InstallationCandidates.emplace_back(D.Dir + "/..");
}

void SYCLInstallationDetector::AddSYCLIncludeArgs(
    const ArgList &DriverArgs, ArgStringList &CC1Args) const {
  // Add the SYCL header search locations in the specified order.
  //   ../include/sycl
  //   ../include/sycl/stl_wrappers
  //   ../include
  SmallString<128> IncludePath(D.Dir);
  llvm::sys::path::append(IncludePath, "..");
  llvm::sys::path::append(IncludePath, "include");
  SmallString<128> SYCLPath(IncludePath);
  llvm::sys::path::append(SYCLPath, "sycl");
  // This is used to provide our wrappers around STL headers that provide
  // additional functions/template specializations when the user includes those
  // STL headers in their programs (e.g., <complex>).
  SmallString<128> STLWrappersPath(SYCLPath);
  llvm::sys::path::append(STLWrappersPath, "stl_wrappers");
  CC1Args.push_back("-internal-isystem");
  CC1Args.push_back(DriverArgs.MakeArgString(SYCLPath));
  CC1Args.push_back("-internal-isystem");
  CC1Args.push_back(DriverArgs.MakeArgString(STLWrappersPath));
  CC1Args.push_back("-internal-isystem");
  CC1Args.push_back(DriverArgs.MakeArgString(IncludePath));
}

void SYCLInstallationDetector::print(llvm::raw_ostream &OS) const {
  if (!InstallationCandidates.size())
    return;
  OS << "SYCL Installation Candidates: \n";
  for (const auto &IC : InstallationCandidates) {
    OS << IC << "\n";
  }
}

// Unsupported options for SYCL device compilation
//  -fcf-protection, -fsanitize, -fprofile-generate, -fprofile-instr-generate
//  -ftest-coverage, -fcoverage-mapping, -fcreate-profile, -fprofile-arcs
//  -fcs-profile-generate -forder-file-instrumentation, --coverage
static std::vector<OptSpecifier> getUnsupportedOpts(void) {
  std::vector<OptSpecifier> UnsupportedOpts = {
      options::OPT_fsanitize_EQ,
      options::OPT_fcf_protection_EQ,
      options::OPT_fprofile_generate,
      options::OPT_fprofile_generate_EQ,
      options::OPT_fno_profile_generate,
      options::OPT_ftest_coverage,
      options::OPT_fno_test_coverage,
      options::OPT_fcoverage_mapping,
      options::OPT_fno_coverage_mapping,
      options::OPT_coverage,
      options::OPT_fprofile_instr_generate,
      options::OPT_fprofile_instr_generate_EQ,
      options::OPT_fprofile_arcs,
      options::OPT_fno_profile_arcs,
      options::OPT_fno_profile_instr_generate,
      options::OPT_fcreate_profile,
      options::OPT_fprofile_instr_use,
      options::OPT_fprofile_instr_use_EQ,
      options::OPT_forder_file_instrumentation,
      options::OPT_fcs_profile_generate,
      options::OPT_fcs_profile_generate_EQ};
  return UnsupportedOpts;
}

SYCLToolChain::SYCLToolChain(const Driver &D, const llvm::Triple &Triple,
                             const ToolChain &HostTC, const ArgList &Args)
    : ToolChain(D, Triple, Args), HostTC(HostTC), SYCLInstallation(D) {
  // Lookup binaries into the driver directory, this is used to discover any
  // dependent SYCL offload compilation tools.
  getProgramPaths().push_back(getDriver().Dir);

  // Diagnose unsupported options only once.
  for (OptSpecifier Opt : getUnsupportedOpts()) {
    if (const Arg *A = Args.getLastArg(Opt)) {
      // All sanitizer options are not currently supported, except
      // AddressSanitizer
      if (A->getOption().getID() == options::OPT_fsanitize_EQ &&
          A->getValues().size() == 1) {
        std::string SanitizeVal = A->getValue();
        if (SanitizeVal == "address")
          continue;
      }
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

void SYCLToolChain::AddSYCLIncludeArgs(const ArgList &DriverArgs,
                                       ArgStringList &CC1Args) const {
  SYCLInstallation.AddSYCLIncludeArgs(DriverArgs, CC1Args);
}

void SYCLToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                              ArgStringList &CC1Args) const {
  HostTC.AddClangSystemIncludeArgs(DriverArgs, CC1Args);
}

void SYCLToolChain::AddClangCXXStdlibIncludeArgs(const ArgList &Args,
                                                 ArgStringList &CC1Args) const {
  HostTC.AddClangCXXStdlibIncludeArgs(Args, CC1Args);
}

SanitizerMask SYCLToolChain::getSupportedSanitizers() const {
  return SanitizerKind::Address;
}
