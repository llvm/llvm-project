//===-- xray_allocator.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// Defines the allocator interface for an arena allocator, used primarily for
// the profiling runtime.
//
//===----------------------------------------------------------------------===//
#ifndef XRAY_ALLOCATOR_H
#define XRAY_ALLOCATOR_H

#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include <cstddef>
#include <cstdint>

#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __xray {

/// The Allocator type hands out fixed-sized chunks of memory that are
/// cache-line aligned and sized. This is useful for placement of
/// performance-sensitive data in memory that's frequently accessed. The
/// allocator also self-limits the peak memory usage to a dynamically defined
/// maximum.
///
/// N is the lower-bound size of the block of memory to return from the
/// allocation function. N is used to compute the size of a block, which is
/// cache-line-size multiples worth of memory. We compute the size of a block by
/// determining how many cache lines worth of memory is required to subsume N.
template <size_t N> struct Allocator {
  // The Allocator returns memory as Block instances.
  struct Block {
    /// Compute the minimum cache-line size multiple that is >= N.
    static constexpr auto Size =
        kCacheLineSize * ((N / kCacheLineSize) + (N % kCacheLineSize ? 1 : 0));
    void *Data = nullptr;
  };

private:
  // A BlockLink will contain a fixed number of blocks, each with an identifier
  // to specify whether it's been handed out or not. We keep track of BlockLink
  // iterators, which are basically a pointer to the link and an offset into
  // the fixed set of blocks associated with a link. The iterators are
  // bidirectional.
  //
  // We're calling it a "link" in the context of seeing these as a chain of
  // block pointer containers (i.e. links in a chain).
  struct BlockLink {
    static_assert(kCacheLineSize % sizeof(void *) == 0,
                  "Cache line size is not divisible by size of void*; none of "
                  "the assumptions of the BlockLink will hold.");

    // We compute the number of pointers to areas in memory where we consider as
    // individual blocks we've allocated. To ensure that instances of the
    // BlockLink object are cache-line sized, we deduct one additional
    // pointers worth representing the pointer to the previous link.
    //
    // This structure corresponds to the following layout:
    //
    //   Blocks [ 0, 1, 2, .., BlockPtrCount - 1]
    //
    static constexpr auto BlockPtrCount =
        (kCacheLineSize / sizeof(Block *)) - 1;

    BlockLink() {
      // Zero out Blocks.
      // FIXME: Use a braced member initializer when we drop support for GCC
      // 4.8.
      internal_memset(Blocks, 0, sizeof(Blocks));
    }

    // FIXME: Align this to cache-line address boundaries?
    Block Blocks[BlockPtrCount];
    BlockLink *Prev = nullptr;
  };

  static_assert(sizeof(BlockLink) == kCacheLineSize,
                "BlockLink instances must be cache-line-sized.");

  static BlockLink NullLink;

  // FIXME: Implement a freelist, in case we actually do intend to return memory
  // to the allocator, as opposed to just de-allocating everything in one go?

  size_t MaxMemory;
  SpinMutex Mutex{};
  BlockLink *Tail = &NullLink;
  size_t Counter = 0;

  BlockLink *NewChainLink() {
    auto NewChain = reinterpret_cast<BlockLink *>(
        InternalAlloc(sizeof(BlockLink), nullptr, kCacheLineSize));
    auto BackingStore = reinterpret_cast<char *>(InternalAlloc(
        BlockLink::BlockPtrCount * Block::Size, nullptr, kCacheLineSize));
    size_t Offset = 0;
    DCHECK_NE(NewChain, nullptr);
    DCHECK_NE(BackingStore, nullptr);
    for (auto &B : NewChain->Blocks) {
      B.Data = BackingStore + Offset;
      Offset += Block::Size;
    }
    NewChain->Prev = Tail;
    return NewChain;
  }

public:
  Allocator(size_t M, size_t PreAllocate) : MaxMemory(M) {
    // FIXME: Implement PreAllocate support!
  }

  Block Allocate() {
    SpinMutexLock Lock(&Mutex);
    // Check whether we're over quota.
    if (Counter * Block::Size >= MaxMemory)
      return {};

    size_t ChainOffset = Counter % BlockLink::BlockPtrCount;

    Block B{};
    BlockLink *Link = Tail;
    if (UNLIKELY(Counter == 0 || ChainOffset == 0))
      Tail = Link = NewChainLink();

    B = Link->Blocks[ChainOffset];
    ++Counter;
    return B;
  }

  ~Allocator() NOEXCEPT {
    // We need to deallocate all the blocks, including the chain links.
    for (auto *C = Tail; C != &NullLink;) {
      // We know that the data block is a large contiguous page, we deallocate
      // that at once.
      InternalFree(C->Blocks[0].Data);
      auto Prev = C->Prev;
      InternalFree(C);
      C = Prev;
    }
  }
}; // namespace __xray

// Storage for the NullLink sentinel.
template <size_t N> typename Allocator<N>::BlockLink Allocator<N>::NullLink;

} // namespace __xray

#endif // XRAY_ALLOCATOR_H
