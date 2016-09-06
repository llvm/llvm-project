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

namespace __scudo {

enum AllocType : u8 {
  FromMalloc    = 0, // Memory block came from malloc, realloc, calloc, etc.
  FromNew       = 1, // Memory block came from operator new.
  FromNewArray  = 2, // Memory block came from operator new [].
  FromMemalign  = 3, // Memory block came from memalign, posix_memalign, etc.
};

struct AllocatorOptions {
  u32 QuarantineSizeMb;
  u32 ThreadLocalQuarantineSizeKb;
  bool MayReturnNull;
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

} // namespace __scudo

#endif  // SCUDO_ALLOCATOR_H_
