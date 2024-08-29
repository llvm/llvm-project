//===--- SYCL.h - SYCL ToolChain Implementations ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SYCL_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SYCL_H

#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"

namespace clang {
namespace driver {

class SYCLInstallationDetector {
public:
  SYCLInstallationDetector(const Driver &D);
  void AddSYCLIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                          llvm::opt::ArgStringList &CC1Args) const;
  void print(llvm::raw_ostream &OS) const;

private:
  const Driver &D;
  llvm::SmallVector<llvm::SmallString<128>, 4> InstallationCandidates;
};

namespace toolchains {

class LLVM_LIBRARY_VISIBILITY SYCLToolChain : public ToolChain {
public:
  SYCLToolChain(const Driver &D, const llvm::Triple &Triple,
                const ToolChain &HostTC, const llvm::opt::ArgList &Args);

  const llvm::Triple *getAuxTriple() const override {
    return &HostTC.getTriple();
  }

  llvm::opt::DerivedArgList *
  TranslateArgs(const llvm::opt::DerivedArgList &Args, StringRef BoundArch,
                Action::OffloadKind DeviceOffloadKind) const override;
  void addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                         llvm::opt::ArgStringList &CC1Args,
                         Action::OffloadKind DeviceOffloadKind) const override;

  bool useIntegratedAs() const override { return true; }
  bool isPICDefault() const override { return false; }
  llvm::codegenoptions::DebugInfoFormat getDefaultDebugFormat() const override {
    if (this->HostTC.getTriple().isWindowsMSVCEnvironment())
      return this->HostTC.getDefaultDebugFormat();
    return ToolChain::getDefaultDebugFormat();
  }
  bool isPIEDefault(const llvm::opt::ArgList &Args) const override {
    return false;
  }
  bool isPICDefaultForced() const override { return false; }

  void addClangWarningOptions(llvm::opt::ArgStringList &CC1Args) const override;
  CXXStdlibType GetCXXStdlibType(const llvm::opt::ArgList &Args) const override;
  void AddSYCLIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                          llvm::opt::ArgStringList &CC1Args) const override;
  void AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;
  void AddClangCXXStdlibIncludeArgs(
      const llvm::opt::ArgList &Args,
      llvm::opt::ArgStringList &CC1Args) const override;

  SanitizerMask getSupportedSanitizers() const override;

  const ToolChain &HostTC;

  SYCLInstallationDetector SYCLInstallation;
};

} // end namespace toolchains

} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SYCL_H
