//===--- HIPAMD.h - HIP ToolChain Implementations ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_HIPAMD_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_HIPAMD_H

#include "AMDGPU.h"
#include "clang/Driver/SyclInstallationDetector.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"

namespace clang {
namespace driver {

namespace tools {

namespace AMDGCN {
// Runs llvm-link/opt/llc/lld, which links multiple LLVM bitcode, together with
// device library, then compiles it to ISA in a shared object.
class LLVM_LIBRARY_VISIBILITY Linker final : public Tool {
public:
  Linker(const ToolChain &TC) : Tool("AMDGCN::Linker", "amdgcn-link", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;

private:
  void constructLldCommand(Compilation &C, const JobAction &JA,
                           const InputInfoList &Inputs, const InputInfo &Output,
                           const llvm::opt::ArgList &Args) const;
  void constructLLVMLinkCommand(Compilation &C, const JobAction &JA,
                                const InputInfoList &Inputs,
                                const InputInfo &Output,
                                const llvm::opt::ArgList &Args) const;
  void constructLinkAndEmitSpirvCommand(Compilation &C, const JobAction &JA,
                                        const InputInfoList &Inputs,
                                        const InputInfo &Output,
                                        const llvm::opt::ArgList &Args) const;
};

} // end namespace AMDGCN
} // end namespace tools

namespace toolchains {

class LLVM_LIBRARY_VISIBILITY SPIRVAMDToolChain final : public AMDGPUToolChain {
public:
  SPIRVAMDToolChain(const Driver &D, const llvm::Triple &Triple,
                    const llvm::opt::ArgList &Args);

  /// SPIR-V uses LTO by default to link device bitcode.
  LTOKind getDefaultLTOMode() const override { return LTOK_Full; }

protected:
  Tool *buildLinker() const override;
};

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_HIPAMD_H
