//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Instrumented heap layout. ABI between the host runtime and the pass.
///
//===----------------------------------------------------------------------===//

#ifndef DASAN_MAPPING_H
#define DASAN_MAPPING_H

#include <stdint.h>

namespace __dasan {

inline constexpr uint64_t kAddressSpaceSize = 1ULL << 47;

inline constexpr uint64_t kSpaceSizeLog = 45;
inline constexpr uint64_t kSpaceSize = 1ULL << kSpaceSizeLog;
inline constexpr uint64_t kSpaceBeg = kSpaceSize;

inline constexpr uint64_t kNumClassesLog = 5;
inline constexpr uint64_t kNumClasses = 1ULL << kNumClassesLog;
inline constexpr uint64_t kRegionSizeLog = kSpaceSizeLog - kNumClassesLog;
inline constexpr uint64_t kRegionSize = 1ULL << kRegionSizeLog;
inline constexpr uint64_t kRegionMask = kRegionSize - 1;

// The device computes the chunk index as offset >> (kMinSizeLog + class).
inline constexpr uint64_t kMinSizeLog = 6;

inline constexpr uint64_t ClassIdToSizeLog(uint64_t ClassId) {
  return kMinSizeLog + ClassId;
}

inline constexpr uint64_t ClassIdToSize(uint64_t ClassId) {
  return 1ULL << ClassIdToSizeLog(ClassId);
}

inline constexpr uint64_t kMetadataSize = 8;

inline constexpr uint64_t ReachableChunks(uint64_t ClassId) {
  return 1ULL << (kRegionSizeLog - ClassIdToSizeLog(ClassId));
}

inline constexpr uint64_t kMetadataBytes = ReachableChunks(0) * kMetadataSize;

inline constexpr uint64_t CountSizeClasses() {
  uint64_t N = 0;
  while (N < kNumClasses && ClassIdToSize(N) <= kRegionSize - kMetadataBytes)
    ++N;
  return N;
}
inline constexpr uint64_t kNumSizeClasses = CountSizeClasses();
inline constexpr uint64_t kSpaceUsed = kNumSizeClasses << kRegionSizeLog;

static_assert(kSpaceUsed == kSpaceBeg,
              "the layout must keep every size class: see above");

inline constexpr uint64_t kMinRedzone = 16;

inline constexpr uint64_t kAllocAlignment = 256;

inline constexpr uint64_t RegionBeg(uint64_t ClassId) {
  return kSpaceBeg + (ClassId << kRegionSizeLog);
}

inline constexpr uint64_t MaxUsableChunks(uint64_t ClassId) {
  return (kRegionSize - kMetadataBytes) >> ClassIdToSizeLog(ClassId);
}

inline constexpr uint64_t kMaxSize =
    ClassIdToSize(kNumSizeClasses - 1) - 2 * kMinRedzone;

// Smallest class whose chunk holds Size plus additional redzoning.
inline constexpr uint64_t ClassID(uint64_t Size) {
  if (Size > kMaxSize)
    return kNumSizeClasses;
  uint64_t ClassId = 0;
  while (ClassIdToSize(ClassId) < Size + 2 * kMinRedzone) ++ClassId;
  return ClassId;
}

// Entry layout.
//   [36:0]   size of the live allocation, cleared on free
//   [37]     poisoned: this chunk held an allocation and it is gone
//   [63:38]  signed offset of the allocation from this chunk's base, in units
//            of the smallest chunk size.
// Every object starts on a class 0 boundary, so scaling the offset costs no
// precision and lets an object span 2 GiB of chunks.
inline constexpr uint64_t kSizeBits = 37;
inline constexpr uint64_t kSizeMask = (1ULL << kSizeBits) - 1;
inline constexpr uint64_t kPoisoned = 1ULL << kSizeBits;
inline constexpr uint64_t kOffsetShift = kSizeBits + 1;
inline constexpr uint64_t kOffsetBits = 64 - kOffsetShift;
inline constexpr uint64_t kOffsetFieldMask = ~((1ULL << kOffsetShift) - 1);
inline constexpr uint64_t kOffsetScaleLog = kMinSizeLog;
inline constexpr int64_t kOffsetScale = 1LL << kOffsetScaleLog;
inline constexpr int64_t kOffsetMax = ((1LL << (kOffsetBits - 1)) - 1)
                                      << kOffsetScaleLog;
inline constexpr int64_t kOffsetMin =
    -((1LL << (kOffsetBits - 1)) << kOffsetScaleLog);

inline constexpr bool OffsetFits(int64_t Offset) {
  return Offset >= kOffsetMin && Offset <= kOffsetMax &&
         Offset % kOffsetScale == 0;
}

inline constexpr uint64_t MakeEntry(uint64_t Size, int64_t Offset) {
  return (Size & kSizeMask) |
         ((static_cast<uint64_t>(Offset) << (kOffsetShift - kOffsetScaleLog)) &
          kOffsetFieldMask);
}

inline constexpr uint64_t EntrySize(uint64_t Entry) {
  return Entry & kSizeMask;
}

inline constexpr int64_t EntryOffset(uint64_t Entry) {
  return static_cast<int64_t>(Entry & kOffsetFieldMask) >>
         (kOffsetShift - kOffsetScaleLog);
}

inline constexpr uint64_t PoisonEntry(uint64_t Entry) {
  return (Entry & ~kSizeMask) | kPoisoned;
}

inline constexpr bool IsPoisoned(uint64_t Entry) { return Entry & kPoisoned; }

inline bool PointerIsMine(uint64_t Addr) {
  return Addr - kSpaceBeg < kSpaceUsed;
}

inline bool PointerIsMetadata(uint64_t Addr) {
  return PointerIsMine(Addr) &&
         (Addr & kRegionMask) >= kRegionSize - kMetadataBytes;
}

inline uint64_t GetSizeClass(uint64_t Addr) {
  return (Addr - kSpaceBeg) >> kRegionSizeLog;
}

inline uint64_t GetChunkIdx(uint64_t Addr, uint64_t ClassId) {
  return (Addr & kRegionMask) >> ClassIdToSizeLog(ClassId);
}

inline constexpr uint64_t GetMetadata(uint64_t Addr, uint64_t ChunkIdx) {
  return (Addr | kRegionMask) - ((ChunkIdx << 3) | 7);
}

inline constexpr bool IsInBounds(uint64_t ChunkOffset, uint64_t Entry) {
  return ChunkOffset - static_cast<uint64_t>(EntryOffset(Entry)) <
         EntrySize(Entry);
}

constexpr bool ValidateSizeClassMap() {
  for (uint64_t C = 0; C < kNumSizeClasses; ++C) {
    if (ClassID(ClassIdToSize(C) - 2 * kMinRedzone) != C)
      return false;
    if (C && ClassID(ClassIdToSize(C - 1) - 2 * kMinRedzone + 1) != C)
      return false;
    if (ClassIdToSize(C) > kRegionSize)
      return false;
    if (MaxUsableChunks(C) == 0)
      return false;
    if (MaxUsableChunks(C) * ClassIdToSize(C) > kRegionSize - kMetadataBytes)
      return false;
    if (ReachableChunks(C) * kMetadataSize > kMetadataBytes)
      return false;
    if (GetMetadata(RegionBeg(C), 0) !=
        RegionBeg(C) + kRegionSize - kMetadataSize)
      return false;
    if (GetMetadata(RegionBeg(C), ReachableChunks(C) - 1) <
        RegionBeg(C) + kRegionSize - kMetadataBytes)
      return false;
  }
  return true;
}

constexpr bool ValidateEntry() {
  uint64_t E = MakeEntry(4000, 128);
  if (EntrySize(E) != 4000 || EntryOffset(E) != 128 || IsPoisoned(E))
    return false;
  if (!IsInBounds(128, E) || !IsInBounds(4127, E))
    return false;
  if (IsInBounds(127, E) || IsInBounds(4128, E) || IsInBounds(0, E))
    return false;
  if (!IsPoisoned(PoisonEntry(E)) || IsInBounds(128, PoisonEntry(E)) ||
      EntryOffset(PoisonEntry(E)) != 128)
    return false;

  const int64_t Back = -static_cast<int64_t>(kAllocAlignment);
  uint64_t S = MakeEntry(1024, Back);
  if (EntryOffset(S) != Back || IsPoisoned(S))
    return false;
  if (!IsInBounds(0, S) || !IsInBounds(255, S))
    return false;
  if (IsInBounds(768, S) || !IsInBounds(767, S))
    return false;

  uint64_t Far = MakeEntry(kSizeMask, kOffsetMin);
  if (EntrySize(Far) != kSizeMask || EntryOffset(Far) != kOffsetMin ||
      IsPoisoned(Far))
    return false;

  return OffsetFits(kOffsetMax) && OffsetFits(kOffsetMin) &&
         !OffsetFits(kOffsetMax + kOffsetScale) &&
         !OffsetFits(kOffsetMin - kOffsetScale) &&
         !OffsetFits(kOffsetScale / 2) &&
         EntryOffset(MakeEntry(1, kOffsetMax)) == kOffsetMax &&
         EntryOffset(MakeEntry(1, kOffsetMin)) == kOffsetMin;
}

static_assert(kSpaceBeg + kSpaceSize <= kAddressSpaceSize,
              "space escapes the 47 bit range");
static_assert(ValidateSizeClassMap(), "size class map does not round trip");
static_assert(ValidateEntry(), "entry encoding does not round trip");
static_assert(kMaxSize <= kSizeMask, "an allocation can outgrow its entry");
static_assert(kAllocAlignment <= kOffsetMax,
              "an offset cannot reach its entry");
static_assert(kMinRedzone <= kAllocAlignment,
              "a redzone is not a whole stride");
static_assert(kAllocAlignment % kOffsetScale == 0,
              "a heap offset is not a whole number of units");
static_assert(ClassIdToSize(0) == kOffsetScale,
              "a global's chunk offset is not a whole number of units");
static_assert(-kOffsetMin >= (1LL << 31), "a large global cannot be described");
static_assert(kSizeBits + kOffsetBits + 1 == 64, "an entry is not one word");

}  // namespace __dasan

#endif  // DASAN_MAPPING_H
