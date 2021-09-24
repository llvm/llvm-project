//===--- M88k.h - M88k-specific Tool Helpers ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_M88K_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_M88K_H

#include "clang/Driver/Driver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/Option.h"
#include <string>
#include <vector>

namespace clang {
namespace driver {
namespace tools {
namespace m88k {

enum class FloatABI {
  Invalid,
  Soft,
  Hard,
};

FloatABI getM88kFloatABI(const Driver &D, const llvm::opt::ArgList &Args);

std::string getM88kTargetCPU(const llvm::opt::ArgList &Args);

void getM88kTargetFeatures(const Driver &D, const llvm::Triple &Triple,
                           const llvm::opt::ArgList &Args,
                           std::vector<llvm::StringRef> &Features);

} // end namespace m88k
} // end namespace tools
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_M88K_H
