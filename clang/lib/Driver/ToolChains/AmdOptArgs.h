//===--- CommonArgs.h - Args handling for multiple toolchains ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_AMDOPTARGS_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_AMDOPTARGS_H

#include "clang/Driver/InputInfo.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Multilib.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/Support/CodeGen.h"

namespace clang {
namespace driver {
namespace tools {

bool checkForAMDProprietaryOptOptions(const ToolChain &TC, const Driver &D,
                                      const llvm::opt::ArgList &Args,
                                      llvm::opt::ArgStringList &CmdArgs,
                                      bool isLLD = false,
				      bool checkOnly = false);

} // end namespace tools
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_COMMONARGS_H
