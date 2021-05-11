//===- OutputSegment.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OutputSegment.h"
#include "InputChunks.h"
#include "lld/Common/Memory.h"

#define DEBUG_TYPE "lld"

using namespace llvm;
using namespace llvm::wasm;

namespace lld {

namespace wasm {

void OutputSegment::addInputSegment(InputSegment *inSeg) {
  alignment = std::max(alignment, inSeg->alignment);
  inputSegments.push_back(inSeg);
  size = llvm::alignTo(size, 1ULL << inSeg->alignment);
  LLVM_DEBUG(dbgs() << "addInputSegment: " << inSeg->getName()
                    << " oname=" << name << " size=" << inSeg->getSize()
                    << " align=" << inSeg->alignment << " at:" << size << "\n");
  inSeg->outputSeg = this;
  inSeg->outputSegmentOffset = size;
  size += inSeg->getSize();
}

// This function scans over the input segments.
//
// It removes MergeInputSegments from the input section array and adds
// new synthetic sections at the location of the first input section
// that it replaces. It then finalizes each synthetic section in order
// to compute an output offset for each piece of each input section.
void OutputSegment::finalizeInputSegments() {
  LLVM_DEBUG(llvm::dbgs() << "finalizeInputSegments: " << name << "\n");
  std::vector<SyntheticMergedDataSegment *> mergedSegments;
  std::vector<InputSegment *> newSegments;
  for (InputSegment *s : inputSegments) {
    MergeInputSegment *ms = dyn_cast<MergeInputSegment>(s);
    if (!ms) {
      newSegments.push_back(s);
      continue;
    }

    // A segment should not make it here unless its alive
    assert(ms->live);

    auto i =
        llvm::find_if(mergedSegments, [=](SyntheticMergedDataSegment *seg) {
          return seg->flags == ms->flags && seg->alignment == ms->alignment;
        });
    if (i == mergedSegments.end()) {
      LLVM_DEBUG(llvm::dbgs() << "new merge section: " << name
                              << " alignment=" << ms->alignment << "\n");
      SyntheticMergedDataSegment *syn =
          make<SyntheticMergedDataSegment>(name, ms->alignment, ms->flags);
      syn->outputSeg = this;
      mergedSegments.push_back(syn);
      i = std::prev(mergedSegments.end());
      newSegments.push_back(syn);
    } else {
      LLVM_DEBUG(llvm::dbgs() << "adding to merge section: " << name << "\n");
    }
    (*i)->addMergeSegment(ms);
  }

  for (auto *ms : mergedSegments)
    ms->finalizeContents();

  inputSegments = newSegments;
  size = 0;
  for (InputSegment *seg : inputSegments) {
    size = llvm::alignTo(size, 1ULL << seg->alignment);
    LLVM_DEBUG(llvm::dbgs() << "outputSegmentOffset set: " << seg->getName()
                            << " -> " << size << "\n");
    seg->outputSegmentOffset = size;
    size += seg->getSize();
  }
}

} // namespace wasm
} // namespace lld
