//===--- LoongArch.h - LoongArch-specific Tool Helpers ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_LOONGARCH_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_LOONGARCH_H

#include "clang/Driver/Driver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/Option.h"

namespace clang {
namespace driver {
namespace tools {
namespace loongarch {
StringRef getLoongArchABI(const llvm::opt::ArgList &Args,
                          const llvm::Triple &Triple);
} // end namespace loongarch
} // end namespace tools
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_LOONGARCH_H
