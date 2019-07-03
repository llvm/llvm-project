//===-- backtrace_linux_libc.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <execinfo.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "gwp_asan/optional/backtrace.h"
#include "gwp_asan/options.h"

namespace {
void Backtrace(uintptr_t *TraceBuffer, size_t Size) {
  // Grab (what seems to be) one more trace than we need. TraceBuffer needs to
  // be null-terminated, but we wish to remove the frame of this function call.
  static_assert(sizeof(uintptr_t) == sizeof(void *), "uintptr_t is not void*");
  int NumTraces =
      backtrace(reinterpret_cast<void **>(TraceBuffer), Size);

  // Now shift the entire trace one place to the left and null-terminate.
  memmove(TraceBuffer, TraceBuffer + 1, NumTraces * sizeof(void *));
  TraceBuffer[NumTraces - 1] = 0;
}

static void PrintBacktrace(uintptr_t *Trace,
                           gwp_asan::options::Printf_t Printf) {
  size_t NumTraces = 0;
  for (; Trace[NumTraces] != 0; ++NumTraces) {
  }

  if (NumTraces == 0) {
    Printf("  <not found (does your allocator support backtracing?)>\n\n");
    return;
  }

  char **BacktraceSymbols =
      backtrace_symbols(reinterpret_cast<void **>(Trace), NumTraces);

  for (size_t i = 0; i < NumTraces; ++i) {
    if (!BacktraceSymbols)
      Printf("  #%zu %p\n", i, Trace[i]);
    else
      Printf("  #%zu %s\n", i, BacktraceSymbols[i]);
  }

  Printf("\n");
  if (BacktraceSymbols)
    free(BacktraceSymbols);
}
} // anonymous namespace

namespace gwp_asan {
namespace options {
Backtrace_t getBacktraceFunction() { return Backtrace; }
PrintBacktrace_t getPrintBacktraceFunction() { return PrintBacktrace; }
} // namespace options
} // namespace gwp_asan
