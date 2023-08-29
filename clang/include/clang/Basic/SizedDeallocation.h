//===------- SizedDellocation.h - Sized Deallocation ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines a function that returns the minimum OS versions supporting
/// C++14's sized deallocation functions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_SIZEDDEALLOCATION_H
#define LLVM_CLANG_BASIC_SIZEDDEALLOCATION_H

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/TargetParser/Triple.h"

namespace clang {
inline llvm::VersionTuple sizedDeallocMinVersion(llvm::Triple::OSType OS) {
  switch (OS) {
  default:
    break;
  case llvm::Triple::Darwin:
  case llvm::Triple::MacOSX: // Earliest supporting version is 10.12.
    return llvm::VersionTuple(10U, 12U);
  case llvm::Triple::IOS:
  case llvm::Triple::TvOS: // Earliest supporting version is 10.0.0.
    return llvm::VersionTuple(10U);
  case llvm::Triple::WatchOS: // Earliest supporting version is 3.0.0.
    return llvm::VersionTuple(3U);
  case llvm::Triple::ZOS:
    return llvm::VersionTuple(); // All z/OS versions have no support.
  }

  llvm_unreachable("Unexpected OS");
}

} // end namespace clang

#endif // LLVM_CLANG_BASIC_SIZEDDEALLOCATION_H
