//===--- SPIRVOpenMP.h - SPIR-V OpenMP Tool Implementations ------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SPIRV_OPENMP_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SPIRV_OPENMP_H

#include "SPIRV.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"

namespace clang::driver::toolchains {
class LLVM_LIBRARY_VISIBILITY SPIRVOpenMPToolChain : public SPIRVToolChain {
public:
  SPIRVOpenMPToolChain(const Driver &D, const llvm::Triple &Triple,
                       const ToolChain &HostTC, const llvm::opt::ArgList &Args);

  void addClangTargetOptions(
      const llvm::opt::ArgList &DriverArgs, llvm::opt::ArgStringList &CC1Args,
      Action::OffloadKind DeviceOffloadingKind) const override;

  const ToolChain &HostTC;
};
} // namespace clang::driver::toolchains
#endif
