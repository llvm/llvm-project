//===--- AArch64.h - AArch64-specific (not ARM) Tool Helpers ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_AARCH64_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_AARCH64_H

#include "clang/Driver/Driver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/Option.h"
#include <string>
#include <vector>

namespace clang {
namespace driver {
namespace tools {
namespace aarch64 {

enum class ReadTPMode {
  Invalid,
  Soft,
  TPIDR_EL0,
  TPIDR_EL1,
  TPIDR_EL2,
  TPIDR_EL3,
  TPIDRRO_EL0,
};

void getAArch64TargetFeatures(const Driver &D, const llvm::Triple &Triple,
                              const llvm::opt::ArgList &Args,
                              std::vector<llvm::StringRef> &Features,
                              bool ForAS, bool ForMultilib = false);

std::string getAArch64TargetCPU(const llvm::opt::ArgList &Args,
                                const llvm::Triple &Triple, llvm::opt::Arg *&A);

ReadTPMode getReadTPMode(const Driver &D, const llvm::opt::ArgList &Args,
                         const llvm::Triple &Triple, bool ForAS);

void setPAuthABIInTriple(const Driver &D, const llvm::opt::ArgList &Args,
                         llvm::Triple &triple);

} // end namespace aarch64
} // end namespace tools
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_AARCH64_H
