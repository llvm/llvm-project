//==- SPIRVOpenMP.cpp - SPIR-V OpenMP Tool Implementations --------*- C++ -*==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==------------------------------------------------------------------------==//
#include "SPIRVOpenMP.h"
#include "CommonArgs.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace llvm::opt;

namespace clang::driver::toolchains {
SPIRVOpenMPToolChain::SPIRVOpenMPToolChain(const Driver &D,
                                           const llvm::Triple &Triple,
                                           const ToolChain &HostToolchain,
                                           const ArgList &Args)
    : SPIRVToolChain(D, Triple, Args), HostTC(HostToolchain) {}

void SPIRVOpenMPToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs, llvm::opt::ArgStringList &CC1Args,
    Action::OffloadKind DeviceOffloadingKind) const {

  if (DeviceOffloadingKind != Action::OFK_OpenMP)
    return;

  if (DriverArgs.hasArg(options::OPT_nogpulib))
    return;
  addOpenMPDeviceRTL(getDriver(), DriverArgs, CC1Args, "", getTriple(), HostTC);
}
} // namespace clang::driver::toolchains
