//===--- Tapir.h - C Language Family Language Options -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Defines helper functions for processing flags related to Tapir.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_TAPIR_H
#define LLVM_CLANG_DRIVER_TAPIR_H

#include "clang/Basic/Tapir.h"

namespace llvm {
namespace opt {
  class ArgList;
}
}

namespace clang {

TapirTargetID parseTapirTarget(const llvm::opt::ArgList &Args);

} // end namespace clang

#endif
