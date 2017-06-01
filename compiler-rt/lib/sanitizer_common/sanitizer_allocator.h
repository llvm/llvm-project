//===-- sanitizer_allocator.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Specialized memory allocator for ThreadSanitizer, MemorySanitizer, etc.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_ALLOCATOR_H
#define SANITIZER_ALLOCATOR_H

#include "sanitizer_internal_defs.h"
#include "sanitizer_common.h"
#include "sanitizer_libc.h"
#include "sanitizer_list.h"
#include "sanitizer_mutex.h"
#include "sanitizer_lfstack.h"
#include "sanitizer_procmaps.h"

namespace __sanitizer {

// Returns true if ReportAllocatorCannotReturnNull(true) was called.
// Can be use to avoid memory hungry operations.
bool IsReportingOOM();

// Prints error message and kills the program.
void NORETURN ReportAllocatorCannotReturnNull(bool out_of_memory);

// Allocators call these callbacks on mmap/munmap.
struct NoOpMapUnmapCallback {
  void OnMap(uptr p, uptr size) const { }
  void OnUnmap(uptr p, uptr size) const { }
};

// Callback type for iterating over chunks.
typedef void (*ForEachChunkCallback)(uptr chunk, void *arg);

// Returns true if calloc(size, n) should return 0 due to overflow in size*n.
bool CallocShouldReturnNullDueToOverflow(uptr size, uptr n);

#include "sanitizer_allocator_size_class_map.h"
#include "sanitizer_allocator_stats.h"
#include "sanitizer_allocator_primary64.h"
#include "sanitizer_allocator_bytemap.h"
#include "sanitizer_allocator_primary32.h"
#include "sanitizer_allocator_local_cache.h"
#include "sanitizer_allocator_secondary.h"
#include "sanitizer_allocator_combined.h"

} // namespace __sanitizer

#endif // SANITIZER_ALLOCATOR_H
