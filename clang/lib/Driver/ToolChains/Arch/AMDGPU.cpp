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
#include "llvm/TargetParser/TargetParser.h"

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

void AMDGPU::setArchNameInTriple(const Driver &D, const ArgList &Args,
                                 types::ID InputType, llvm::Triple &Triple) {
  StringRef MArch;
  AMDGPU::getAMDGPUArchCPUFromArgs(Triple, Args, MArch);

  if (MArch == "amdgcnspirv") {
    Triple.setArch(llvm::Triple::ArchType::spirv64);
    return;
  }
}

void AMDGPU::getAMDGPUArchCPUFromArgs(const llvm::Triple &Triple,
                                      const llvm::opt::ArgList &Args,
                                      llvm::StringRef &Arch) {
  if (const Arg *MArch = Args.getLastArg(options::OPT_march_EQ))
    Arch = MArch->getValue();
  else if (const Arg *MCPU = Args.getLastArg(options::OPT_mcpu_EQ))
    Arch = MCPU->getValue();
}
