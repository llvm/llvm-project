//===-- scudo_allocator.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// Header for scudo_allocator.cpp.
///
//===----------------------------------------------------------------------===//

#ifndef SCUDO_ALLOCATOR_H_
#define SCUDO_ALLOCATOR_H_

#ifndef __x86_64__
# error "The Scudo hardened allocator currently only supports x86_64."
#endif

#include "scudo_flags.h"

#include "sanitizer_common/sanitizer_allocator.h"

#include <atomic>

namespace __scudo {

enum AllocType : u8 {
  FromMalloc    = 0, // Memory block came from malloc, realloc, calloc, etc.
  FromNew       = 1, // Memory block came from operator new.
  FromNewArray  = 2, // Memory block came from operator new [].
  FromMemalign  = 3, // Memory block came from memalign, posix_memalign, etc.
};

enum ChunkState : u8 {
  ChunkAvailable  = 0,
  ChunkAllocated  = 1,
  ChunkQuarantine = 2
};

#if SANITIZER_WORDSIZE == 64
// Our header requires 128 bits of storage on 64-bit platforms, which fits
// nicely with the alignment requirements. Having the offset saves us from
// using functions such as GetBlockBegin, that is fairly costly. Our first
// implementation used the MetaData as well, which offers the advantage of
// being stored away from the chunk itself, but accessing it was costly as
// well. The header will be atomically loaded and stored using the 16-byte
// primitives offered by the platform (likely requires cmpxchg16b support).
typedef unsigned __int128 PackedHeader;
struct UnpackedHeader {
  u16  Checksum      : 16;
  uptr RequestedSize : 40; // Needed for reallocation purposes.
  u8   State         : 2;  // available, allocated, or quarantined
  u8   AllocType     : 2;  // malloc, new, new[], or memalign
  u8   Unused_0_     : 4;
  uptr Offset        : 12; // Offset from the beginning of the backend
                           // allocation to the beginning of the chunk itself,
                           // in multiples of MinAlignment. See comment about
                           // its maximum value and test in init().
  u64  Unused_1_     : 36;
  u16  Salt          : 16;
};
#elif SANITIZER_WORDSIZE == 32
// On 32-bit platforms, our header requires 64 bits.
typedef u64 PackedHeader;
struct UnpackedHeader {
  u16  Checksum      : 12;
  uptr RequestedSize : 32; // Needed for reallocation purposes.
  u8   State         : 2;  // available, allocated, or quarantined
  u8   AllocType     : 2;  // malloc, new, new[], or memalign
  uptr Offset        : 12; // Offset from the beginning of the backend
                           // allocation to the beginning of the chunk itself,
                           // in multiples of MinAlignment. See comment about
                           // its maximum value and test in Allocator::init().
  u16  Salt          : 4;
};
#else
# error "Unsupported SANITIZER_WORDSIZE."
#endif  // SANITIZER_WORDSIZE

typedef std::atomic<PackedHeader> AtomicPackedHeader;
COMPILER_CHECK(sizeof(UnpackedHeader) == sizeof(PackedHeader));

const uptr ChunkHeaderSize = sizeof(PackedHeader);

// Minimum alignment of 8 bytes for 32-bit, 16 for 64-bit
const uptr MinAlignmentLog = FIRST_32_SECOND_64(3, 4);
const uptr MaxAlignmentLog = 24; // 16 MB
const uptr MinAlignment = 1 << MinAlignmentLog;
const uptr MaxAlignment = 1 << MaxAlignmentLog;

struct AllocatorOptions {
  u32 QuarantineSizeMb;
  u32 ThreadLocalQuarantineSizeKb;
  bool MayReturnNull;
  s32 ReleaseToOSIntervalMs;
  bool DeallocationTypeMismatch;
  bool DeleteSizeMismatch;
  bool ZeroContents;

  void setFrom(const Flags *f, const CommonFlags *cf);
  void copyTo(Flags *f, CommonFlags *cf) const;
};

void initAllocator(const AllocatorOptions &options);
void drainQuarantine();

void *scudoMalloc(uptr Size, AllocType Type);
void scudoFree(void *Ptr, AllocType Type);
void scudoSizedFree(void *Ptr, uptr Size, AllocType Type);
void *scudoRealloc(void *Ptr, uptr Size);
void *scudoCalloc(uptr NMemB, uptr Size);
void *scudoMemalign(uptr Alignment, uptr Size);
void *scudoValloc(uptr Size);
void *scudoPvalloc(uptr Size);
int scudoPosixMemalign(void **MemPtr, uptr Alignment, uptr Size);
void *scudoAlignedAlloc(uptr Alignment, uptr Size);
uptr scudoMallocUsableSize(void *Ptr);

#include "scudo_allocator_secondary.h"

} // namespace __scudo

#endif  // SCUDO_ALLOCATOR_H_
