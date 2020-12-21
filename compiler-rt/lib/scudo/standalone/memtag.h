//===-- memtag.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_MEMTAG_H_
#define SCUDO_MEMTAG_H_

#include "internal_defs.h"

#if SCUDO_LINUX
#include <sys/auxv.h>
#include <sys/prctl.h>
#endif

namespace scudo {

void setRandomTag(void *Ptr, uptr Size, uptr ExcludeMask, uptr *TaggedBegin,
                  uptr *TaggedEnd);

#if defined(__aarch64__) || defined(SCUDO_FUZZ)

inline constexpr bool archSupportsMemoryTagging() { return true; }
inline constexpr uptr archMemoryTagGranuleSize() { return 16; }

inline uptr untagPointer(uptr Ptr) { return Ptr & ((1ULL << 56) - 1); }

inline uint8_t extractTag(uptr Ptr) { return (Ptr >> 56) & 0xf; }

#else

inline constexpr bool archSupportsMemoryTagging() { return false; }

inline uptr archMemoryTagGranuleSize() {
  UNREACHABLE("memory tagging not supported");
}

inline uptr untagPointer(uptr Ptr) {
  (void)Ptr;
  UNREACHABLE("memory tagging not supported");
}

inline uint8_t extractTag(uptr Ptr) {
  (void)Ptr;
  UNREACHABLE("memory tagging not supported");
}

#endif

#if defined(__aarch64__)

inline bool systemSupportsMemoryTagging() {
#ifndef HWCAP2_MTE
#define HWCAP2_MTE (1 << 18)
#endif
  return getauxval(AT_HWCAP2) & HWCAP2_MTE;
}

inline bool systemDetectsMemoryTagFaultsTestOnly() {
#ifndef PR_GET_TAGGED_ADDR_CTRL
#define PR_GET_TAGGED_ADDR_CTRL 56
#endif
#ifndef PR_MTE_TCF_SHIFT
#define PR_MTE_TCF_SHIFT 1
#endif
#ifndef PR_MTE_TCF_NONE
#define PR_MTE_TCF_NONE (0UL << PR_MTE_TCF_SHIFT)
#endif
#ifndef PR_MTE_TCF_MASK
#define PR_MTE_TCF_MASK (3UL << PR_MTE_TCF_SHIFT)
#endif
  return (static_cast<unsigned long>(
              prctl(PR_GET_TAGGED_ADDR_CTRL, 0, 0, 0, 0)) &
          PR_MTE_TCF_MASK) != PR_MTE_TCF_NONE;
}

inline void disableMemoryTagChecksTestOnly() {
  __asm__ __volatile__(".arch_extension mte; msr tco, #1");
}

inline void enableMemoryTagChecksTestOnly() {
  __asm__ __volatile__(".arch_extension mte; msr tco, #0");
}

class ScopedDisableMemoryTagChecks {
  size_t PrevTCO;

public:
  ScopedDisableMemoryTagChecks() {
    __asm__ __volatile__(".arch_extension mte; mrs %0, tco; msr tco, #1"
                         : "=r"(PrevTCO));
  }

  ~ScopedDisableMemoryTagChecks() {
    __asm__ __volatile__(".arch_extension mte; msr tco, %0" : : "r"(PrevTCO));
  }
};

inline uptr selectRandomTag(uptr Ptr, uptr ExcludeMask) {
  uptr TaggedPtr;
  __asm__ __volatile__(
      ".arch_extension mte; irg %[TaggedPtr], %[Ptr], %[ExcludeMask]"
      : [TaggedPtr] "=r"(TaggedPtr)
      : [Ptr] "r"(Ptr), [ExcludeMask] "r"(ExcludeMask));
  return TaggedPtr;
}

inline uptr storeTags(uptr Begin, uptr End) {
  DCHECK(Begin % 16 == 0);
  if (Begin != End) {
    __asm__ __volatile__(
        R"(
      .arch_extension mte

    1:
      stzg %[Cur], [%[Cur]], #16
      cmp %[Cur], %[End]
      b.lt 1b
    )"
        : [Cur] "+&r"(Begin)
        : [End] "r"(End)
        : "memory");
  }
  return Begin;
}

inline void *prepareTaggedChunk(void *Ptr, uptr Size, uptr ExcludeMask,
                                uptr BlockEnd) {
  // Prepare the granule before the chunk to store the chunk header by setting
  // its tag to 0. Normally its tag will already be 0, but in the case where a
  // chunk holding a low alignment allocation is reused for a higher alignment
  // allocation, the chunk may already have a non-zero tag from the previous
  // allocation.
  __asm__ __volatile__(".arch_extension mte; stg %0, [%0, #-16]"
                       :
                       : "r"(Ptr)
                       : "memory");

  uptr TaggedBegin, TaggedEnd;
  setRandomTag(Ptr, Size, ExcludeMask, &TaggedBegin, &TaggedEnd);

  // Finally, set the tag of the granule past the end of the allocation to 0,
  // to catch linear overflows even if a previous larger allocation used the
  // same block and tag. Only do this if the granule past the end is in our
  // block, because this would otherwise lead to a SEGV if the allocation
  // covers the entire block and our block is at the end of a mapping. The tag
  // of the next block's header granule will be set to 0, so it will serve the
  // purpose of catching linear overflows in this case.
  uptr UntaggedEnd = untagPointer(TaggedEnd);
  if (UntaggedEnd != BlockEnd)
    __asm__ __volatile__(".arch_extension mte; stg %0, [%0]"
                         :
                         : "r"(UntaggedEnd)
                         : "memory");
  return reinterpret_cast<void *>(TaggedBegin);
}

inline void resizeTaggedChunk(uptr OldPtr, uptr NewPtr, uptr BlockEnd) {
  uptr RoundOldPtr = roundUpTo(OldPtr, 16);
  if (RoundOldPtr >= NewPtr) {
    // If the allocation is shrinking we just need to set the tag past the end
    // of the allocation to 0. See explanation in prepareTaggedChunk above.
    uptr RoundNewPtr = untagPointer(roundUpTo(NewPtr, 16));
    if (RoundNewPtr != BlockEnd)
      __asm__ __volatile__(".arch_extension mte; stg %0, [%0]"
                           :
                           : "r"(RoundNewPtr)
                           : "memory");
    return;
  }

  __asm__ __volatile__(R"(
    .arch_extension mte

    // Set the memory tag of the region
    // [roundUpTo(OldPtr, 16), roundUpTo(NewPtr, 16))
    // to the pointer tag stored in OldPtr.
  1:
    stzg %[Cur], [%[Cur]], #16
    cmp %[Cur], %[End]
    b.lt 1b

    // Finally, set the tag of the granule past the end of the allocation to 0.
    and %[Cur], %[Cur], #(1 << 56) - 1
    cmp %[Cur], %[BlockEnd]
    b.eq 2f
    stg %[Cur], [%[Cur]]

  2:
  )"
                       : [Cur] "+&r"(RoundOldPtr), [End] "+&r"(NewPtr)
                       : [BlockEnd] "r"(BlockEnd)
                       : "memory");
}

inline uptr loadTag(uptr Ptr) {
  uptr TaggedPtr = Ptr;
  __asm__ __volatile__(".arch_extension mte; ldg %0, [%0]"
                       : "+r"(TaggedPtr)
                       :
                       : "memory");
  return TaggedPtr;
}

#else

inline bool systemSupportsMemoryTagging() {
  UNREACHABLE("memory tagging not supported");
}

inline bool systemDetectsMemoryTagFaultsTestOnly() {
  UNREACHABLE("memory tagging not supported");
}

inline void disableMemoryTagChecksTestOnly() {
  UNREACHABLE("memory tagging not supported");
}

inline void enableMemoryTagChecksTestOnly() {
  UNREACHABLE("memory tagging not supported");
}

struct ScopedDisableMemoryTagChecks {
  ScopedDisableMemoryTagChecks() {}
};

inline uptr selectRandomTag(uptr Ptr, uptr ExcludeMask) {
  (void)Ptr;
  (void)ExcludeMask;
  UNREACHABLE("memory tagging not supported");
}

inline uptr storeTags(uptr Begin, uptr End) {
  (void)Begin;
  (void)End;
  UNREACHABLE("memory tagging not supported");
}

inline void *prepareTaggedChunk(void *Ptr, uptr Size, uptr ExcludeMask,
                                uptr BlockEnd) {
  (void)Ptr;
  (void)Size;
  (void)ExcludeMask;
  (void)BlockEnd;
  UNREACHABLE("memory tagging not supported");
}

inline void resizeTaggedChunk(uptr OldPtr, uptr NewPtr, uptr BlockEnd) {
  (void)OldPtr;
  (void)NewPtr;
  (void)BlockEnd;
  UNREACHABLE("memory tagging not supported");
}

inline uptr loadTag(uptr Ptr) {
  (void)Ptr;
  UNREACHABLE("memory tagging not supported");
}

#endif

inline void setRandomTag(void *Ptr, uptr Size, uptr ExcludeMask,
                         uptr *TaggedBegin, uptr *TaggedEnd) {
  *TaggedBegin = selectRandomTag(reinterpret_cast<uptr>(Ptr), ExcludeMask);
  *TaggedEnd = storeTags(*TaggedBegin, *TaggedBegin + Size);
}

} // namespace scudo

#endif
