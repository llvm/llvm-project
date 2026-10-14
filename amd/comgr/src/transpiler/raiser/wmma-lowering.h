//===- wmma-lowering.h - Transpiler matrix remapping -----------*- C++ -*-===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_WMMA_LOWERING_H
#define HOTSWAP_TRANSPILER_WMMA_LOWERING_H

#include "llvm/Support/Error.h"

namespace llvm {
class Value;
}

namespace COMGR::transpiler {

class RaiseContext;

namespace detail {

struct FragmentDwordLocation {
  unsigned Lane;
  unsigned Dword;
};

constexpr FragmentDwordLocation getWMMAInputDword(unsigned SourceWave,
                                                  unsigned TargetLane,
                                                  unsigned TargetDword,
                                                  unsigned KHalf,
                                                  unsigned DwordsPerLane) {
  unsigned LaneGroup = TargetLane / 16;
  return {SourceWave * 32 + (LaneGroup / 2) * 16 + TargetLane % 16,
          (KHalf * 2 + LaneGroup % 2) * DwordsPerLane + TargetDword};
}

constexpr FragmentDwordLocation getWMMAAccumulatorDword(unsigned SourceWave,
                                                        unsigned TargetLane,
                                                        unsigned TargetDword) {
  unsigned LaneGroup = TargetLane / 16;
  return {SourceWave * 32 + (LaneGroup / 2) * 16 + TargetLane % 16,
          (LaneGroup % 2) * 4 + TargetDword};
}

constexpr FragmentDwordLocation getMFMAResultDword(unsigned SourceLane,
                                                   unsigned SourceDword) {
  return {(SourceLane / 16) * 32 + (SourceDword / 4) * 16 + SourceLane % 16,
          SourceDword % 4};
}

} // namespace detail

enum class WMMAInputType {
  F32,
  F16,
  BF16,
  IU8,
};

/// Remap one wave32 16x16 WMMA fragment to wave64 MFMA layout.
llvm::Expected<llvm::Value *> emitWMMAtoMFMA(RaiseContext &Ctx, llvm::Value *A,
                                             llvm::Value *B, llvm::Value *C,
                                             WMMAInputType InputType);

} // namespace COMGR::transpiler

#endif
