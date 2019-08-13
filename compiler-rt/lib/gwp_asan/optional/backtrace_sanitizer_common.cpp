//===-- backtrace_sanitizer_common.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "gwp_asan/optional/backtrace.h"
#include "gwp_asan/options.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

void __sanitizer::BufferedStackTrace::UnwindImpl(uptr pc, uptr bp,
                                                 void *context,
                                                 bool request_fast,
                                                 u32 max_depth) {
  if (!StackTrace::WillUseFastUnwind(request_fast)) {
    return Unwind(max_depth, pc, bp, context, 0, 0, request_fast);
  }
  Unwind(max_depth, pc, 0, context, 0, 0, false);
}

namespace {
size_t Backtrace(uintptr_t *TraceBuffer, size_t Size) {
  __sanitizer::BufferedStackTrace Trace;
  Trace.Reset();
  if (Size > __sanitizer::kStackTraceMax)
    Size = __sanitizer::kStackTraceMax;

  Trace.Unwind((__sanitizer::uptr)__builtin_return_address(0),
               (__sanitizer::uptr)__builtin_frame_address(0),
               /* ucontext */ nullptr,
               /* fast unwind */ true, Size - 1);

  memcpy(TraceBuffer, Trace.trace, Trace.size * sizeof(uintptr_t));
  return Trace.size;
}

static void PrintBacktrace(uintptr_t *Trace, size_t TraceLength,
                           gwp_asan::options::Printf_t Printf) {
  __sanitizer::StackTrace StackTrace;
  StackTrace.trace = reinterpret_cast<__sanitizer::uptr *>(Trace);
  StackTrace.size = TraceLength;

  if (StackTrace.size == 0) {
    Printf("  <unknown (does your allocator support backtracing?)>\n\n");
    return;
  }

  StackTrace.Print();
}
} // anonymous namespace

namespace gwp_asan {
namespace options {
Backtrace_t getBacktraceFunction() { return Backtrace; }
PrintBacktrace_t getPrintBacktraceFunction() { return PrintBacktrace; }
} // namespace options
} // namespace gwp_asan
