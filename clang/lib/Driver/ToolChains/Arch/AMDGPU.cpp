//===--- AMDGPU.cpp - AMDGPU Helpers for Tools ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "clang/Basic/TargetID.h"
#include "clang/Driver/Driver.h"
#include "clang/Options/Options.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

void AMDGPU::setArchNameInTriple(const Driver &D, const ArgList &Args,
                                 BoundArch BA, types::ID InputType,
                                 llvm::Triple &Triple) {
  StringRef MArch = BA.ArchName;
  if (MArch.empty())
    AMDGPU::getAMDGPUArchCPUFromArgs(Triple, Args, MArch);

  if (MArch == "amdgcnspirv") {
    Triple.setArch(llvm::Triple::ArchType::spirv64);
    return;
  }

  StringRef ProcName = getProcessorFromTargetID(Triple, MArch);
  llvm::AMDGPU::GPUKind GK = llvm::AMDGPU::parseArchAMDGCN(ProcName);
  if (GK == llvm::AMDGPU::GPUKind::GK_NONE) {
    // Normalize legacy "amdgcn" triples to "amdgpu"
    Triple.setArch(Triple.getArch(), Triple.getSubArch());
    return;
  }

  Triple.setArch(Triple.getArch(), static_cast<llvm::Triple::SubArchType>(
                                       llvm::AMDGPU::getSubArch(GK)));
}

void AMDGPU::getAMDGPUArchCPUFromArgs(const llvm::Triple &Triple,
                                      const llvm::opt::ArgList &Args,
                                      llvm::StringRef &Arch) {
  if (const Arg *MCPU = Args.getLastArg(options::OPT_mcpu_EQ))
    Arch = MCPU->getValue();
}
