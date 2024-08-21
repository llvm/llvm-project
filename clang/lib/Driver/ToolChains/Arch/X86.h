//===--- X86.h - X86-specific Tool Helpers ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_X86_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_X86_H

#include "clang/Driver/Driver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/Option.h"
#include "llvm/TargetParser/Triple.h"
#include <string>
#include <vector>

namespace clang {
namespace driver {
namespace tools {

enum class DecimalFloatABI {
  Default,    // Target-specific default.
  None,       // No decimal floating-point support.
  Libgcc_BID, // libgcc with BID encoding.
  Libgcc_DPD, // libgcc with DPD encoding.
  Hard        // Target-specific hard ABI.
};
DecimalFloatABI getDecimalFloatABI(const Driver &D,
                                      const llvm::Triple &Triple,
                                      const llvm::opt::ArgList &Args);

DecimalFloatABI getDefaultDecimalFloatABI(const llvm::Triple &Triple);

namespace x86 {

std::string getX86TargetCPU(const Driver &D, const llvm::opt::ArgList &Args,
                            const llvm::Triple &Triple);

void getX86TargetFeatures(const Driver &D, const llvm::Triple &Triple,
                          const llvm::opt::ArgList &Args,
                          std::vector<llvm::StringRef> &Features);

} // end namespace x86
} // end namespace target
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_X86_H
