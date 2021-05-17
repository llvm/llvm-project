//===- AMDGPUOpenMP.h - AMDGPUOpenMP ToolChain Implementation -*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_AMDGPUOPENMP_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_AMDGPUOPENMP_H

#include "AMDGPU.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"

namespace clang {
namespace driver {

namespace tools {

namespace AMDGCN {
// Runs llvm-link/opt/llc/lld, which links multiple LLVM bitcode, together with
// device library, then compiles it to ISA in a shared object.
class LLVM_LIBRARY_VISIBILITY OpenMPLinker : public Tool {
public:
  OpenMPLinker(const ToolChain &TC)
      : Tool("AMDGCN::OpenMPLinker", "amdgcn-link", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;

private:
  /// \return llvm-link output file name.
  const char *constructLLVMLinkCommand(Compilation &C, const JobAction &JA,
                                       const InputInfoList &Inputs,
                                       const llvm::opt::ArgList &Args,
                                       llvm::StringRef SubArchName,
                                       llvm::StringRef OutputFilePrefix) const;

  /// \return llc output file name.
  const char *constructLlcCommand(Compilation &C, const JobAction &JA,
                                  const InputInfoList &Inputs,
                                  const llvm::opt::ArgList &Args,
                                  llvm::StringRef SubArchName,
                                  llvm::StringRef OutputFilePrefix,
                                  const char *InputFileName,
                                  bool OutputIsAsm = false) const;

  void constructLldCommand(Compilation &C, const JobAction &JA,
                           const InputInfoList &Inputs, const InputInfo &Output,
                           const llvm::opt::ArgList &Args,
                           const char *InputFileName) const;
};

} // end namespace AMDGCN
} // end namespace tools

namespace toolchains {

class LLVM_LIBRARY_VISIBILITY AMDGPUOpenMPToolChain final
    : public ROCMToolChain {
public:
  AMDGPUOpenMPToolChain(const Driver &D, const llvm::Triple &Triple,
                        const ToolChain &HostTC,
                        const llvm::opt::ArgList &Args);

  const llvm::Triple *getAuxTriple() const override {
    return &HostTC.getTriple();
  }

  llvm::opt::DerivedArgList *
  TranslateArgs(const llvm::opt::DerivedArgList &Args, StringRef BoundArch,
                Action::OffloadKind DeviceOffloadKind) const override;
  void
  addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                        llvm::opt::ArgStringList &CC1Args,
                        Action::OffloadKind DeviceOffloadKind) const override;
  void addClangWarningOptions(llvm::opt::ArgStringList &CC1Args) const override;
  CXXStdlibType GetCXXStdlibType(const llvm::opt::ArgList &Args) const override;
  void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;
  void AddIAMCUIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                           llvm::opt::ArgStringList &CC1Args) const override;

  SanitizerMask getSupportedSanitizers() const override;

  VersionTuple
  computeMSVCVersion(const Driver *D,
                     const llvm::opt::ArgList &Args) const override;

  const ToolChain &HostTC;

protected:
  Tool *buildLinker() const override;
};

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_AMDGPUOPENMP_H
