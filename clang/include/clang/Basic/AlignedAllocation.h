//===--- AlignedAllocation.h - Aligned Allocation ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines a function that returns the minimum OS versions supporting
/// C++17's aligned allocation functions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_ALIGNED_ALLOCATION_H
#define LLVM_CLANG_BASIC_ALIGNED_ALLOCATION_H

#include "clang/Basic/VersionTuple.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang {

inline VersionTuple alignedAllocMinVersion(llvm::Triple::OSType OS) {
  switch (OS) {
  default:
    break;
  case llvm::Triple::Darwin:
  case llvm::Triple::MacOSX: // Earliest supporting version is 10.13.
    return VersionTuple(10U, 13U);
  case llvm::Triple::IOS:
  case llvm::Triple::TvOS: // Earliest supporting version is 11.0.0.
    return VersionTuple(11U);
  case llvm::Triple::WatchOS: // Earliest supporting version is 4.0.0.
    return VersionTuple(4U);
  }

  llvm_unreachable("Unexpected OS");
}

} // end namespace clang

#endif // LLVM_CLANG_BASIC_ALIGNED_ALLOCATION_H
