//===- bolt/Passes/StackPointerTracking.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the StackPointerTracking class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/StackPointerTracking.h"

namespace llvm {
namespace bolt {

StackPointerTracking::StackPointerTracking(
    BinaryFunction &BF, MCPlusBuilder::AllocatorIdTy AllocatorId)
    : StackPointerTrackingBase<StackPointerTracking>(BF, AllocatorId) {}

} // end namespace bolt
} // end namespace llvm

llvm::raw_ostream &llvm::operator<<(llvm::raw_ostream &OS,
                                    const std::pair<int, int> &Val) {
  OS << Val.first << ", " << Val.second;
  return OS;
}
