//===-- hwasan.h --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

typedef u8 tag_t;

// Reasonable values are 4 (for 1/16th shadow) and 6 (for 1/64th).
const uptr kShadowScale = 4;
const uptr kShadowAlignment = 1UL << kShadowScale;

#define MEM_TO_SHADOW_OFFSET(mem) ((uptr)(mem) >> kShadowScale)
#define MEM_TO_SHADOW(mem) ((uptr)(mem) >> kShadowScale)
#define SHADOW_TO_MEM(shadow) ((uptr)(shadow) << kShadowScale)

#define MEM_IS_APP(mem) MemIsApp((uptr)(mem))

// TBI (Top Byte Ignore) feature of AArch64: bits [63:56] are ignored in address
// translation and can be used to store a tag.
const unsigned kAddressTagShift = 56;
const uptr kAddressTagMask = 0xFFUL << kAddressTagShift;

static inline tag_t GetTagFromPointer(uptr p) {
  return p >> kAddressTagShift;
}

static inline uptr GetAddressFromPointer(uptr p) {
  return p & ~kAddressTagMask;
}

static inline void * GetAddressFromPointer(const void *p) {
  return (void *)((uptr)p & ~kAddressTagMask);
}

static inline uptr AddTagToPointer(uptr p, tag_t tag) {
  return (p & ~kAddressTagMask) | ((uptr)tag << kAddressTagShift);
}

namespace __hwasan {

extern int hwasan_inited;
extern bool hwasan_init_is_running;
extern int hwasan_report_count;

bool MemIsApp(uptr p);

bool ProtectRange(uptr beg, uptr end);
bool InitShadow();
char *GetProcSelfMaps();
void InitializeInterceptors();

void HwasanAllocatorInit();
void HwasanAllocatorThreadFinish();
void HwasanDeallocate(StackTrace *stack, void *ptr);

void *hwasan_malloc(uptr size, StackTrace *stack);
void *hwasan_calloc(uptr nmemb, uptr size, StackTrace *stack);
void *hwasan_realloc(void *ptr, uptr size, StackTrace *stack);
void *hwasan_valloc(uptr size, StackTrace *stack);
void *hwasan_pvalloc(uptr size, StackTrace *stack);
void *hwasan_aligned_alloc(uptr alignment, uptr size, StackTrace *stack);
void *hwasan_memalign(uptr alignment, uptr size, StackTrace *stack);
int hwasan_posix_memalign(void **memptr, uptr alignment, uptr size,
                        StackTrace *stack);

void InstallTrapHandler();
void InstallAtExitHandler();

const char *GetStackOriginDescr(u32 id, uptr *pc);

void EnterSymbolizer();
void ExitSymbolizer();
bool IsInSymbolizer();

struct SymbolizerScope {
  SymbolizerScope() { EnterSymbolizer(); }
  ~SymbolizerScope() { ExitSymbolizer(); }
};

void PrintWarning(uptr pc, uptr bp);

void GetStackTrace(BufferedStackTrace *stack, uptr max_s, uptr pc, uptr bp,
                   void *context, bool request_fast_unwind);

void ReportInvalidAccess(StackTrace *stack, u32 origin);
void ReportTagMismatch(StackTrace *stack, uptr addr, uptr access_size,
                       bool is_store);
void ReportStats();
void ReportAtExitStatistics();
void DescribeMemoryRange(const void *x, uptr size);
void ReportInvalidAccessInsideAddressRange(const char *what, const void *start, uptr size,
                                 uptr offset);

// Returns a "chained" origin id, pointing to the given stack trace followed by
// the previous origin id.
u32 ChainOrigin(u32 id, StackTrace *stack);

const int STACK_TRACE_TAG_POISON = StackTrace::TAG_CUSTOM + 1;

#define GET_MALLOC_STACK_TRACE                                            \
  BufferedStackTrace stack;                                               \
  if (hwasan_inited)                                                       \
  GetStackTrace(&stack, common_flags()->malloc_context_size,              \
                StackTrace::GetCurrentPc(), GET_CURRENT_FRAME(), nullptr, \
                common_flags()->fast_unwind_on_malloc)

#define GET_FATAL_STACK_TRACE_PC_BP(pc, bp)              \
  BufferedStackTrace stack;                              \
  if (hwasan_inited)                                       \
  GetStackTrace(&stack, kStackTraceMax, pc, bp, nullptr, \
                common_flags()->fast_unwind_on_fatal)

class ScopedThreadLocalStateBackup {
 public:
  ScopedThreadLocalStateBackup() { Backup(); }
  ~ScopedThreadLocalStateBackup() { Restore(); }
  void Backup();
  void Restore();
 private:
  u64 va_arg_overflow_size_tls;
};

void HwasanTSDInit(void (*destructor)(void *tsd));
void *HwasanTSDGet();
void HwasanTSDSet(void *tsd);
void HwasanTSDDtor(void *tsd);

void HwasanOnDeadlySignal(int signo, void *info, void *context);

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
