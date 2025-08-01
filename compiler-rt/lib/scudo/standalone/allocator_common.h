//===-- allocator_common.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_ALLOCATOR_COMMON_H_
#define SCUDO_ALLOCATOR_COMMON_H_

#include "common.h"
#include "list.h"

namespace scudo {

template <class SizeClassAllocator> struct Batch {
  typedef typename SizeClassAllocator::SizeClassMap SizeClassMap;
  typedef typename SizeClassAllocator::CompactPtrT CompactPtrT;

  void setFromArray(CompactPtrT *Array, u16 N) {
    DCHECK_LE(N, SizeClassAllocator::MaxNumBlocksInBatch);
    Count = N;
    memcpy(Blocks, Array, sizeof(Blocks[0]) * Count);
  }
  void appendFromArray(CompactPtrT *Array, u16 N) {
    DCHECK_LE(N, SizeClassAllocator::MaxNumBlocksInBatch - Count);
    memcpy(Blocks + Count, Array, sizeof(Blocks[0]) * N);
    // u16 will be promoted to int by arithmetic type conversion.
    Count = static_cast<u16>(Count + N);
  }
  void appendFromBatch(Batch *B, u16 N) {
    DCHECK_LE(N, SizeClassAllocator::MaxNumBlocksInBatch - Count);
    DCHECK_GE(B->Count, N);
    // Append from the back of `B`.
    memcpy(Blocks + Count, B->Blocks + (B->Count - N), sizeof(Blocks[0]) * N);
    // u16 will be promoted to int by arithmetic type conversion.
    Count = static_cast<u16>(Count + N);
    B->Count = static_cast<u16>(B->Count - N);
  }
  void clear() { Count = 0; }
  bool empty() { return Count == 0; }
  void add(CompactPtrT P) {
    DCHECK_LT(Count, SizeClassAllocator::MaxNumBlocksInBatch);
    Blocks[Count++] = P;
  }
  void moveToArray(CompactPtrT *Array) {
    memcpy(Array, Blocks, sizeof(Blocks[0]) * Count);
    clear();
  }

  void moveNToArray(CompactPtrT *Array, u16 N) {
    DCHECK_LE(N, Count);
    memcpy(Array, Blocks + Count - N, sizeof(Blocks[0]) * N);
    Count = static_cast<u16>(Count - N);
  }
  u16 getCount() const { return Count; }
  bool isEmpty() const { return Count == 0U; }
  CompactPtrT get(u16 I) const {
    DCHECK_LE(I, Count);
    return Blocks[I];
  }
  Batch *Next;

private:
  u16 Count;
  CompactPtrT Blocks[];
};

// A BatchGroup is used to collect blocks. Each group has a group id to
// identify the group kind of contained blocks.
template <class SizeClassAllocator> struct BatchGroup {
  // `Next` is used by IntrusiveList.
  BatchGroup *Next;
  // The compact base address of each group
  uptr CompactPtrGroupBase;
  // This is used to track how many bytes are not in-use since last time we
  // tried to release pages.
  uptr BytesInBGAtLastCheckpoint;
  // Blocks are managed by Batch in a list.
  SinglyLinkedList<Batch<SizeClassAllocator>> Batches;
  // Cache value of SizeClassAllocatorLocalCache::getMaxCached()
  // TODO(chiahungduan): Except BatchClass, every Batch stores the same number
  // of blocks. As long as we make BatchClass follow this constraint, this
  // field can be removed.
  u16 MaxCachedPerBatch;
};

} // namespace scudo

#endif // SCUDO_ALLOCATOR_COMMON_H_
