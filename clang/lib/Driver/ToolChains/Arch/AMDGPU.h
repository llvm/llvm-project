//===--- AMDGPU.h - AMDGPU-specific Tool Helpers ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_AMDGPU_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_AMDGPU_H

#include "clang/Driver/ToolChain.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include <string>

namespace clang {
namespace driver {
namespace tools {
namespace AMDGPU {

void setArchNameInTriple(const Driver &D, const llvm::opt::ArgList &Args,
                         types::ID InputType, llvm::Triple &Triple);
void getAMDGPUArchCPUFromArgs(const llvm::Triple &Triple,
                              const llvm::opt::ArgList &Args,
                              llvm::StringRef &Arch);
} // end namespace AMDGPU
} // end namespace tools
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_AMDGPU_H
