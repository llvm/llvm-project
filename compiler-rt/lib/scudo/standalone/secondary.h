//===-- secondary.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_SECONDARY_H_
#define SCUDO_SECONDARY_H_

#include "common.h"
#include "mutex.h"
#include "stats.h"

namespace scudo {

// This allocator wraps the platform allocation primitives, and as such is on
// the slower side and should preferably be used for larger sized allocations.
// Blocks allocated will be preceded and followed by a guard page, and hold
// their own header that is not checksummed: the guard pages and the Combined
// header should be enough for our purpose.

namespace LargeBlock {

struct Header {
  LargeBlock::Header *Prev;
  LargeBlock::Header *Next;
  uptr BlockEnd;
  uptr MapBase;
  uptr MapSize;
  MapPlatformData Data;
};

constexpr uptr getHeaderSize() {
  return roundUpTo(sizeof(Header), 1U << SCUDO_MIN_ALIGNMENT_LOG);
}

static Header *getHeader(uptr Ptr) {
  return reinterpret_cast<Header *>(Ptr - getHeaderSize());
}

static Header *getHeader(const void *Ptr) {
  return getHeader(reinterpret_cast<uptr>(Ptr));
}

} // namespace LargeBlock

class MapAllocator {
public:
  void initLinkerInitialized(GlobalStats *S) {
    Stats.initLinkerInitialized();
    if (LIKELY(S))
      S->link(&Stats);
  }
  void init(GlobalStats *S) {
    memset(this, 0, sizeof(*this));
    initLinkerInitialized(S);
  }

  void *allocate(uptr Size, uptr AlignmentHint = 0, uptr *BlockEnd = nullptr);

  void deallocate(void *Ptr);

  static uptr getBlockEnd(void *Ptr) {
    return LargeBlock::getHeader(Ptr)->BlockEnd;
  }

  static uptr getBlockSize(void *Ptr) {
    return getBlockEnd(Ptr) - reinterpret_cast<uptr>(Ptr);
  }

  void printStats() const;

  void disable() { Mutex.lock(); }

  void enable() { Mutex.unlock(); }

  template <typename F> void iterateOverBlocks(F Callback) const {
    for (LargeBlock::Header *H = Tail; H != nullptr; H = H->Prev)
      Callback(reinterpret_cast<uptr>(H) + LargeBlock::getHeaderSize());
  }

private:
  HybridMutex Mutex;
  LargeBlock::Header *Tail;
  uptr AllocatedBytes;
  uptr FreedBytes;
  uptr LargestSize;
  u32 NumberOfAllocs;
  u32 NumberOfFrees;
  LocalStats Stats;
};

} // namespace scudo

#endif // SCUDO_SECONDARY_H_
