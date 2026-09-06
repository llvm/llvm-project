//=-- lsan.h --------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer.
// Private header for standalone LSan RTL.
//
//===----------------------------------------------------------------------===//

#include "lsan_thread.h"
#if SANITIZER_POSIX
#  include "lsan_posix.h"
#elif SANITIZER_FUCHSIA
#  include "lsan_fuchsia.h"
#endif
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

// Unwinds from an explicit frame. The free interceptors capture their stack in
// an out-of-line helper (see lsan_allocator.h) and pass their own pc and bp
// along, so that the reported stack starts at the intercepted function and not
// at the helper.
#define GET_STACK_TRACE_AT(pc, bp, max_size, fast) \
  __sanitizer::BufferedStackTrace stack;           \
  stack.Unwind((pc), (bp), nullptr, fast, max_size);

#define GET_STACK_TRACE(max_size, fast)                               \
  GET_STACK_TRACE_AT(StackTrace::GetCurrentPc(), GET_CURRENT_FRAME(), \
                     max_size, fast)

#define GET_STACK_TRACE_FATAL \
  GET_STACK_TRACE(kStackTraceMax, common_flags()->fast_unwind_on_fatal)

#define GET_STACK_TRACE_MALLOC                         \
  GET_STACK_TRACE(common_flags()->malloc_context_size, \
                  common_flags()->fast_unwind_on_malloc)

// Like GET_STACK_TRACE_MALLOC, but unwinds from an explicit frame.
#define GET_STACK_TRACE_FREE_AT(pc, bp)                   \
  GET_STACK_TRACE_AT((pc), (bp),                          \
                     common_flags()->malloc_context_size, \
                     common_flags()->fast_unwind_on_malloc)

#define GET_STACK_TRACE_THREAD GET_STACK_TRACE(kStackTraceMax, true)

namespace __lsan {

void InitializeInterceptors();
void ReplaceSystemMalloc();
void LsanOnDeadlySignal(int signo, void *siginfo, void *context);
void InstallAtExitCheckLeaks();
void InstallAtForkHandler();

#define ENSURE_LSAN_INITED        \
  do {                            \
    CHECK(!lsan_init_is_running); \
    if (!lsan_inited)             \
      __lsan_init();              \
  } while (0)

}  // namespace __lsan

extern bool lsan_inited;
extern bool lsan_init_is_running;

extern "C" SANITIZER_INTERFACE_ATTRIBUTE void __lsan_init();
