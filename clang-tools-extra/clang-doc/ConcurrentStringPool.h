//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines ConcurrentStringPool, a lock-free deduplicating string
/// saver for clang-doc.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_CONCURRENTSTRINGPOOL_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_CONCURRENTSTRINGPOOL_H

#include "llvm/ADT/LazyAtomicPointer.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/PerThreadBumpPtrAllocator.h"
#include "llvm/Support/xxhash.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

namespace clang {
namespace doc {

/// A thread-safe string pool that deduplicates strings into a
/// PerThreadBumpPtrAllocator.
///
/// Keys are sharded by the upper bits of their hash. Each shard holds an array
/// with a power-of-two number of slabs which are allocated on demand via
/// LazyAtomicPointer. Lookups probe a small window in each slab and move to
/// the next level when the window is full, avoiding locks and table rehashing.
///
/// \tparam ShardBits Number of upper hash bits used to select a shard (1..31).
/// \tparam InitialSlabCapacity Power-of-two capacity of level 0.
/// \tparam ProbeWindow Linear probe window per level (1..InitialSlabCapacity).
/// \tparam NumLevels Number of doubling slab levels per shard; the largest
///   slab (InitialSlabCapacity << (NumLevels - 1)) must not exceed
///   2^(32 - ShardBits), the bits of ProbeHash not consumed by ShardBits.
template <size_t ShardBits = 10, uint32_t InitialSlabCapacity = 256,
          uint32_t ProbeWindow = 8, size_t NumLevels = 15>
class ConcurrentStringPoolImpl {
  static_assert(ShardBits > 0 && ShardBits < 32,
                "ShardBits must be between 1 and 31");
  static_assert(llvm::isPowerOf2_32(InitialSlabCapacity),
                "InitialSlabCapacity must be a power of two");
  static_assert(ProbeWindow > 0 && ProbeWindow <= InitialSlabCapacity,
                "ProbeWindow must fit within InitialSlabCapacity");
  static_assert(NumLevels > 0 && NumLevels < 32,
                "NumLevels must be between 1 and 31");

  static constexpr size_t NumShards = 1ULL << ShardBits;
  static constexpr size_t ProbeBits = 32 - ShardBits;
  static constexpr uint64_t MaxSlabCapacity = 1ULL << ProbeBits;
  static constexpr uint64_t LargestSlabCapacity = uint64_t(InitialSlabCapacity)
                                                  << (NumLevels - 1);
  static constexpr size_t CacheLineBytes = 64;

  static_assert(LargestSlabCapacity <= MaxSlabCapacity,
                "Slab capacity must not overlap ShardBits");

  // Store Size and Hash alongside Data so probe mismatches don't chase the
  // string pointer into the arena.
  struct Bucket {
    llvm::LazyAtomicPointer<const char> Data;
    uint32_t Size = 0;
    uint32_t Hash = 0;
  };
  static_assert(sizeof(Bucket) <= CacheLineBytes / 4,
                "a cache line should fit at least 4 Buckets");

  // Align shards to cache lines to avoid false sharing.
  struct alignas(CacheLineBytes) Shard {
    std::array<llvm::LazyAtomicPointer<Bucket>, NumLevels> Slabs{};

    ~Shard() {
      for (llvm::LazyAtomicPointer<Bucket> &SlabPtr : Slabs)
        delete[] SlabPtr.load();
    }
  };

public:
  ConcurrentStringPoolImpl() = default;
  ~ConcurrentStringPoolImpl() = default;

  ConcurrentStringPoolImpl(const ConcurrentStringPoolImpl &) = delete;
  ConcurrentStringPoolImpl &
  operator=(const ConcurrentStringPoolImpl &) = delete;

  /// Save string \p S in the arena.
  /// \return a reference to the interned string.
  llvm::StringRef save(const char *S) { return save(llvm::StringRef(S)); }

  llvm::StringRef save(const std::string &S) {
    return save(llvm::StringRef(S));
  }

  llvm::StringRef save(const llvm::Twine &S) {
    if (S.isTriviallyEmpty())
      return llvm::StringRef();
    if (S.isSingleStringRef())
      return save(S.getSingleStringRef());
    return saveTwineSlow(S);
  }

  llvm::StringRef save(llvm::StringRef S) {
    if (S.empty())
      return llvm::StringRef();
    return insert(S);
  }

private:
  LLVM_ATTRIBUTE_NOINLINE llvm::StringRef saveTwineSlow(const llvm::Twine &S) {
    llvm::SmallString<128> Storage;
    S.toVector(Storage);
    return save(llvm::StringRef(Storage.data(), Storage.size()));
  }

  llvm::StringRef insert(llvm::StringRef S) {
    assert(S.size() <= std::numeric_limits<uint32_t>::max() &&
           "String exceeds 32-bit size limit");

    // Hash the string and select the shard.
    uint64_t Hash = llvm::xxh3_64bits(S);
    uint32_t HashTag = static_cast<uint32_t>(Hash);
    uint32_t ProbeHash = static_cast<uint32_t>(Hash >> 32);
    Shard &Sh = (*Shards)[Hash >> (64 - ShardBits)];

    // Find the right bucket in the shard to intern into via probing.
    for (size_t Level = 0; Level < NumLevels; ++Level) {
      uint32_t SlabCapacity = InitialSlabCapacity << Level;
      Bucket *CurSlab = &Sh.Slabs[Level].loadOrGenerate(
          [SlabCapacity] { return new Bucket[SlabCapacity](); });
      uint32_t Mask = SlabCapacity - 1;
      uint32_t StartIdx = ProbeHash & Mask;
      for (uint32_t P = 0; P < ProbeWindow; ++P) {
        Bucket &B = CurSlab[(StartIdx + P) & Mask];
        const char &First = B.Data.loadOrGenerate([&] {
          B.Size = static_cast<uint32_t>(S.size());
          B.Hash = HashTag;
          return S.copy(Allocator).data();
        });
        llvm::StringRef Entry(&First, B.Size);
        if (LLVM_LIKELY(B.Hash == HashTag && Entry == S))
          return Entry;
      }
    }
    llvm_unreachable("ConcurrentStringPool capacity exhausted");
  }

  using ShardArray = std::array<Shard, NumShards>;
  std::unique_ptr<ShardArray> Shards = std::make_unique<ShardArray>();
  llvm::parallel::PerThreadBumpPtrAllocator Allocator;
};

using ConcurrentStringPool = ConcurrentStringPoolImpl<>;

} // namespace doc
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_CONCURRENTSTRINGPOOL_H
