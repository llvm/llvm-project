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

#include "hwasan_flags.h"
#include "hwasan_interface_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "ubsan/ubsan_platform.h"

#ifndef HWASAN_CONTAINS_UBSAN
# define HWASAN_CONTAINS_UBSAN CAN_SANITIZE_UB
#endif

#ifndef HWASAN_WITH_INTERCEPTORS
#define HWASAN_WITH_INTERCEPTORS 0
#endif

#ifndef HWASAN_REPLACE_OPERATORS_NEW_AND_DELETE
#define HWASAN_REPLACE_OPERATORS_NEW_AND_DELETE HWASAN_WITH_INTERCEPTORS
#endif

typedef u8 tag_t;

#if defined(__x86_64__)
// Tags are done in middle bits using userspace aliasing.
constexpr unsigned kAddressTagShift = 39;
constexpr unsigned kTagBits = 3;

// The alias region is placed next to the shadow so the upper bits of all
// taggable addresses matches the upper bits of the shadow base.  This shift
// value determines which upper bits must match.  It has a floor of 44 since the
// shadow is always 8TB.
// TODO(morehouse): In alias mode we can shrink the shadow and use a
// simpler/faster shadow calculation.
constexpr unsigned kTaggableRegionCheckShift =
    __sanitizer::Max(kAddressTagShift + kTagBits + 1U, 44U);
#else
// TBI (Top Byte Ignore) feature of AArch64: bits [63:56] are ignored in address
// translation and can be used to store a tag.
constexpr unsigned kAddressTagShift = 56;
constexpr unsigned kTagBits = 8;
#endif  // defined(__x86_64__)

// Mask for extracting tag bits from the lower 8 bits.
constexpr uptr kTagMask = (1UL << kTagBits) - 1;

// Mask for extracting tag bits from full pointers.
constexpr uptr kAddressTagMask = kTagMask << kAddressTagShift;

// Minimal alignment of the shadow base address. Determines the space available
// for threads and stack histories. This is an ABI constant.
const unsigned kShadowBaseAlignment = 32;

const unsigned kRecordAddrBaseTagShift = 3;
const unsigned kRecordFPShift = 48;
const unsigned kRecordFPLShift = 4;
const unsigned kRecordFPModulus = 1 << (64 - kRecordFPShift + kRecordFPLShift);

static inline tag_t GetTagFromPointer(uptr p) {
  return (p >> kAddressTagShift) & kTagMask;
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

bool InitShadow();
void InitPrctl();
void InitThreads();
void InitializeInterceptors();

void HwasanAllocatorInit();

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

void InstallAtExitHandler();

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

#if HWASAN_WITH_INTERCEPTORS && defined(__aarch64__)
// For both bionic and glibc __sigset_t is an unsigned long.
typedef unsigned long __hw_sigset_t;
// Setjmp and longjmp implementations are platform specific, and hence the
// interception code is platform specific too.  As yet we've only implemented
// the interception for AArch64.
typedef unsigned long long __hw_register_buf[22];
struct __hw_jmp_buf_struct {
  // NOTE: The machine-dependent definition of `__sigsetjmp'
  // assume that a `__hw_jmp_buf' begins with a `__hw_register_buf' and that
  // `__mask_was_saved' follows it.  Do not move these members or add others
  // before it.
  __hw_register_buf __jmpbuf; // Calling environment.
  int __mask_was_saved;       // Saved the signal mask?
  __hw_sigset_t __saved_mask; // Saved signal mask.
};
typedef struct __hw_jmp_buf_struct __hw_jmp_buf[1];
typedef struct __hw_jmp_buf_struct __hw_sigjmp_buf[1];
#endif // HWASAN_WITH_INTERCEPTORS && __aarch64__

#endif  // HWASAN_H
