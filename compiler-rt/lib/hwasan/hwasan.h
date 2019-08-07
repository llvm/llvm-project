//===-- hwasan.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of HWAddressSanitizer.
//
// Private Hwasan header.
//===----------------------------------------------------------------------===//

#ifndef HWASAN_H
#define HWASAN_H

#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "hwasan_interface_internal.h"
#include "hwasan_flags.h"
#include "ubsan/ubsan_platform.h"

#ifndef HWASAN_REPLACE_OPERATORS_NEW_AND_DELETE
# define HWASAN_REPLACE_OPERATORS_NEW_AND_DELETE 1
#endif

#ifndef HWASAN_CONTAINS_UBSAN
# define HWASAN_CONTAINS_UBSAN CAN_SANITIZE_UB
#endif

#ifndef HWASAN_WITH_INTERCEPTORS
#define HWASAN_WITH_INTERCEPTORS 0
#endif

typedef u8 tag_t;

// TBI (Top Byte Ignore) feature of AArch64: bits [63:56] are ignored in address
// translation and can be used to store a tag.
const unsigned kAddressTagShift = 56;
const uptr kAddressTagMask = 0xFFUL << kAddressTagShift;

// Minimal alignment of the shadow base address. Determines the space available
// for threads and stack histories. This is an ABI constant.
const unsigned kShadowBaseAlignment = 32;

const unsigned kRecordAddrBaseTagShift = 3;
const unsigned kRecordFPShift = 48;
const unsigned kRecordFPLShift = 4;
const unsigned kRecordFPModulus = 1 << (64 - kRecordFPShift + kRecordFPLShift);

static inline tag_t GetTagFromPointer(uptr p) {
  return p >> kAddressTagShift;
}

static inline uptr UntagAddr(uptr tagged_addr) {
  return tagged_addr & ~kAddressTagMask;
}

static inline void *UntagPtr(const void *tagged_ptr) {
  return reinterpret_cast<void *>(
      UntagAddr(reinterpret_cast<uptr>(tagged_ptr)));
}

static inline uptr AddTagToPointer(uptr p, tag_t tag) {
  return (p & ~kAddressTagMask) | ((uptr)tag << kAddressTagShift);
}

namespace __hwasan {

extern int hwasan_inited;
extern bool hwasan_init_is_running;
extern int hwasan_report_count;

bool ProtectRange(uptr beg, uptr end);
bool InitShadow();
void InitThreads();
void MadviseShadow();
char *GetProcSelfMaps();
void InitializeInterceptors();

void HwasanAllocatorInit();
void HwasanAllocatorThreadFinish();

void *hwasan_malloc(uptr size, StackTrace *stack);
void *hwasan_calloc(uptr nmemb, uptr size, StackTrace *stack);
void *hwasan_realloc(void *ptr, uptr size, StackTrace *stack);
void *hwasan_reallocarray(void *ptr, uptr nmemb, uptr size, StackTrace *stack);
void *hwasan_valloc(uptr size, StackTrace *stack);
void *hwasan_pvalloc(uptr size, StackTrace *stack);
void *hwasan_aligned_alloc(uptr alignment, uptr size, StackTrace *stack);
void *hwasan_memalign(uptr alignment, uptr size, StackTrace *stack);
int hwasan_posix_memalign(void **memptr, uptr alignment, uptr size,
                        StackTrace *stack);
void hwasan_free(void *ptr, StackTrace *stack);

void InstallTrapHandler();
void InstallAtExitHandler();

void EnterSymbolizer();
void ExitSymbolizer();
bool IsInSymbolizer();

struct SymbolizerScope {
  SymbolizerScope() { EnterSymbolizer(); }
  ~SymbolizerScope() { ExitSymbolizer(); }
};

// Returns a "chained" origin id, pointing to the given stack trace followed by
// the previous origin id.
u32 ChainOrigin(u32 id, StackTrace *stack);

const int STACK_TRACE_TAG_POISON = StackTrace::TAG_CUSTOM + 1;

#define GET_MALLOC_STACK_TRACE                                            \
  BufferedStackTrace stack;                                               \
  if (hwasan_inited)                                                      \
    stack.Unwind(StackTrace::GetCurrentPc(), GET_CURRENT_FRAME(),         \
                 nullptr, common_flags()->fast_unwind_on_malloc,          \
                 common_flags()->malloc_context_size)

#define GET_FATAL_STACK_TRACE_PC_BP(pc, bp)              \
  BufferedStackTrace stack;                              \
  if (hwasan_inited)                                     \
    stack.Unwind(pc, bp, nullptr, common_flags()->fast_unwind_on_fatal)

#define GET_FATAL_STACK_TRACE_HERE \
  GET_FATAL_STACK_TRACE_PC_BP(StackTrace::GetCurrentPc(), GET_CURRENT_FRAME())

#define PRINT_CURRENT_STACK_CHECK() \
  {                                 \
    GET_FATAL_STACK_TRACE_HERE;     \
    stack.Print();                  \
  }

class ScopedThreadLocalStateBackup {
 public:
  ScopedThreadLocalStateBackup() { Backup(); }
  ~ScopedThreadLocalStateBackup() { Restore(); }
  void Backup();
  void Restore();
 private:
  u64 va_arg_overflow_size_tls;
};

void HwasanTSDInit();
void HwasanTSDThreadInit();

void HwasanOnDeadlySignal(int signo, void *info, void *context);

void UpdateMemoryUsage();

void AppendToErrorMessageBuffer(const char *buffer);

void AndroidTestTlsSlot();

}  // namespace __hwasan

#define HWASAN_MALLOC_HOOK(ptr, size)       \
  do {                                    \
    if (&__sanitizer_malloc_hook) {       \
      __sanitizer_malloc_hook(ptr, size); \
    }                                     \
    RunMallocHooks(ptr, size);            \
  } while (false)
#define HWASAN_FREE_HOOK(ptr)       \
  do {                            \
    if (&__sanitizer_free_hook) { \
      __sanitizer_free_hook(ptr); \
    }                             \
    RunFreeHooks(ptr);            \
  } while (false)

#endif  // HWASAN_H
