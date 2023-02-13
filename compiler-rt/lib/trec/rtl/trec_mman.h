//===-- trec_mman.h ---------------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of TraceRecorder (TRec), a race detector.
//
//===----------------------------------------------------------------------===//
#ifndef TREC_MMAN_H
#define TREC_MMAN_H

#include "trec_defs.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __trec {

const __sanitizer::uptr kDefaultAlignment = 16;

void InitializeAllocator();
void InitializeAllocatorLate();
void ReplaceSystemMalloc();
void AllocatorProcStart(Processor *proc);
void AllocatorProcFinish(Processor *proc);
void AllocatorPrintStats();

// For user allocations.
void *user_alloc_internal(ThreadState *thr, __sanitizer::uptr pc, __sanitizer::uptr sz,
                          __sanitizer::uptr align = kDefaultAlignment, bool signal = true,
                          bool trace_record = false,
                          const char *func = nullptr);
// Does not accept NULL.
void user_free(ThreadState *thr, __sanitizer::uptr pc, void *p, bool signal = true,
               bool record_trace = false);
// Interceptor implementations.
void *user_alloc(ThreadState *thr, __sanitizer::uptr pc, __sanitizer::uptr sz);
void *user_calloc(ThreadState *thr, __sanitizer::uptr pc, __sanitizer::uptr sz, __sanitizer::uptr n);
void *user_realloc(ThreadState *thr, __sanitizer::uptr pc, void *p, __sanitizer::uptr sz);
void *user_reallocarray(ThreadState *thr, __sanitizer::uptr pc, void *p, __sanitizer::uptr sz, __sanitizer::uptr n);
void *user_memalign(ThreadState *thr, __sanitizer::uptr pc, __sanitizer::uptr align, __sanitizer::uptr sz);
int user_posix_memalign(ThreadState *thr, __sanitizer::uptr pc, void **memptr, __sanitizer::uptr align,
                        __sanitizer::uptr sz);
void *user_aligned_alloc(ThreadState *thr, __sanitizer::uptr pc, __sanitizer::uptr align, __sanitizer::uptr sz);
void *user_valloc(ThreadState *thr, __sanitizer::uptr pc, __sanitizer::uptr sz);
void *user_pvalloc(ThreadState *thr, __sanitizer::uptr pc, __sanitizer::uptr sz);
__sanitizer::uptr user_alloc_usable_size(void *p);

// Invoking malloc/free hooks that may be installed by the user.
void invoke_malloc_hook(void *ptr, __sanitizer::uptr size);
void invoke_free_hook(void *ptr);

enum MBlockType {
  MBlockScopedBuf,
  MBlockString,
  MBlockStackTrace,
  MBlockShadowStack,
  MBlockSync,
  MBlockClock,
  MBlockThreadContex,
  MBlockDeadInfo,
  MBlockRacyStacks,
  MBlockRacyAddresses,
  MBlockAtExit,
  MBlockFlag,
  MBlockReport,
  MBlockReportMop,
  MBlockReportThread,
  MBlockReportMutex,
  MBlockReportLoc,
  MBlockReportStack,
  MBlockSuppression,
  MBlockExpectRace,
  MBlockSignal,
  MBlockJmpBuf,

  // This must be the last.
  MBlockTypeCount
};

// For internal data structures.
void *internal_alloc(MBlockType typ, __sanitizer::uptr sz);
void internal_free(void *p);

template <typename T>
void DestroyAndFree(T *p) {
  p->~T();
  internal_free(p);
}

}  // namespace __trec
#endif  // TREC_MMAN_H
