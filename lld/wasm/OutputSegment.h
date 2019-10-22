//===- OutputSegment.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_WASM_OUTPUT_SEGMENT_H
#define LLD_WASM_OUTPUT_SEGMENT_H

#include "InputChunks.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/Object/Wasm.h"

namespace lld {
namespace wasm {

class InputSegment;

class OutputSegment {
public:
  OutputSegment(StringRef N, uint32_t Index) : Name(N), Index(Index) {}

  void addInputSegment(InputSegment *InSeg) {
    Alignment = std::max(Alignment, InSeg->getAlignment());
    InputSegments.push_back(InSeg);
    Size = llvm::alignTo(Size, 1ULL << InSeg->getAlignment());
    InSeg->OutputSeg = this;
    InSeg->OutputSegmentOffset = Size;
    Size += InSeg->getSize();
  }

  StringRef Name;
  const uint32_t Index;
  uint32_t SectionOffset = 0;
  uint32_t Alignment = 0;
  uint32_t StartVA = 0;
  std::vector<InputSegment *> InputSegments;

  // Sum of the size of the all the input segments
  uint32_t Size = 0;

  // Segment header
  std::string Header;
};

} // namespace wasm
} // namespace lld

#endif // LLD_WASM_OUTPUT_SEGMENT_H
