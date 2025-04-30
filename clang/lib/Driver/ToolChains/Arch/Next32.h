//===--- Next32.h - Next32-specific Tool Helpers ----------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_NEXT32_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_NEXT32_H

#include "clang/Driver/Driver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/Option.h"
#include "llvm/TargetParser/Triple.h"
#include <string>
#include <vector>

namespace clang {
namespace driver {
namespace tools {
namespace next32 {

void getNext32TargetFeatures(const Driver &D, const llvm::Triple &Triple,
                             const llvm::opt::ArgList &Args,
                             std::vector<llvm::StringRef> &Features);

} // end namespace next32
} // namespace tools
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_NEXT32_H
