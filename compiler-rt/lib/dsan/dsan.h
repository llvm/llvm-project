//=-- dsan.h --------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DoubleFreeSanitizer.
// Private header for standalone DSan RTL.
//
//===----------------------------------------------------------------------===//

#include "dsan_thread.h"
#if SANITIZER_POSIX
#  include "dsan_posix.h"
#elif SANITIZER_FUCHSIA
#  include "dsan_fuchsia.h"
#endif
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

#define GET_STACK_TRACE(max_size, fast)                                        \
  __sanitizer::BufferedStackTrace stack;                                       \
  stack.Unwind(StackTrace::GetCurrentPc(), GET_CURRENT_FRAME(), nullptr, fast, \
               max_size);

#define GET_STACK_TRACE_FATAL \
  GET_STACK_TRACE(kStackTraceMax, common_flags()->fast_unwind_on_fatal)

#define GET_STACK_TRACE_MALLOC                                      \
  GET_STACK_TRACE(__sanitizer::common_flags()->malloc_context_size, \
                  common_flags()->fast_unwind_on_malloc)

#define GET_STACK_TRACE_THREAD GET_STACK_TRACE(kStackTraceMax, true)

namespace __dsan {

void InitializeInterceptors();
void ReplaceSystemMalloc();
void DsanOnDeadlySignal(int signo, void* siginfo, void* context);
void InstallAtForkHandler();

#define ENSURE_DSAN_INITED        \
  do {                            \
    CHECK(!dsan_init_is_running); \
    if (!dsan_inited)             \
      __dsan_init();              \
  } while (0)

}  // namespace __dsan

extern bool dsan_inited;
extern bool dsan_init_is_running;

extern "C" SANITIZER_INTERFACE_ATTRIBUTE void __dsan_init();
