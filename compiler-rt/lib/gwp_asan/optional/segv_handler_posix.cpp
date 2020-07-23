//===-- crash_handler_posix.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gwp_asan/common.h"
#include "gwp_asan/crash_handler.h"
#include "gwp_asan/guarded_pool_allocator.h"
#include "gwp_asan/optional/segv_handler.h"
#include "gwp_asan/options.h"

#include <assert.h>
#include <inttypes.h>
#include <signal.h>
#include <stdio.h>

namespace {
using gwp_asan::AllocationMetadata;
using gwp_asan::Error;
using gwp_asan::GuardedPoolAllocator;
using gwp_asan::crash_handler::PrintBacktrace_t;
using gwp_asan::crash_handler::Printf_t;
using gwp_asan::crash_handler::SegvBacktrace_t;

struct sigaction PreviousHandler;
bool SignalHandlerInstalled;
gwp_asan::GuardedPoolAllocator *GPAForSignalHandler;
Printf_t PrintfForSignalHandler;
PrintBacktrace_t PrintBacktraceForSignalHandler;
SegvBacktrace_t BacktraceForSignalHandler;

static void sigSegvHandler(int sig, siginfo_t *info, void *ucontext) {
  if (GPAForSignalHandler) {
    GPAForSignalHandler->stop();

    gwp_asan::crash_handler::dumpReport(
        reinterpret_cast<uintptr_t>(info->si_addr),
        GPAForSignalHandler->getAllocatorState(),
        GPAForSignalHandler->getMetadataRegion(), BacktraceForSignalHandler,
        PrintfForSignalHandler, PrintBacktraceForSignalHandler, ucontext);
  }

  // Process any previous handlers.
  if (PreviousHandler.sa_flags & SA_SIGINFO) {
    PreviousHandler.sa_sigaction(sig, info, ucontext);
  } else if (PreviousHandler.sa_handler == SIG_DFL) {
    // If the previous handler was the default handler, cause a core dump.
    signal(SIGSEGV, SIG_DFL);
    raise(SIGSEGV);
  } else if (PreviousHandler.sa_handler == SIG_IGN) {
    // If the previous segv handler was SIGIGN, crash iff we were responsible
    // for the crash.
    if (__gwp_asan_error_is_mine(GPAForSignalHandler->getAllocatorState(),
                                 reinterpret_cast<uintptr_t>(info->si_addr))) {
      signal(SIGSEGV, SIG_DFL);
      raise(SIGSEGV);
    }
  } else {
    PreviousHandler.sa_handler(sig);
  }
}

struct ScopedEndOfReportDecorator {
  ScopedEndOfReportDecorator(gwp_asan::crash_handler::Printf_t Printf)
      : Printf(Printf) {}
  ~ScopedEndOfReportDecorator() { Printf("*** End GWP-ASan report ***\n"); }
  gwp_asan::crash_handler::Printf_t Printf;
};

// Prints the provided error and metadata information.
void printHeader(Error E, uintptr_t AccessPtr,
                 const gwp_asan::AllocationMetadata *Metadata,
                 Printf_t Printf) {
  // Print using intermediate strings. Platforms like Android don't like when
  // you print multiple times to the same line, as there may be a newline
  // appended to a log file automatically per Printf() call.
  constexpr size_t kDescriptionBufferLen = 128;
  char DescriptionBuffer[kDescriptionBufferLen] = "";
  if (E != Error::UNKNOWN && Metadata != nullptr) {
    uintptr_t Address = __gwp_asan_get_allocation_address(Metadata);
    size_t Size = __gwp_asan_get_allocation_size(Metadata);
    if (E == Error::USE_AFTER_FREE) {
      snprintf(DescriptionBuffer, kDescriptionBufferLen,
               "(%zu byte%s into a %zu-byte allocation at 0x%zx) ",
               AccessPtr - Address, (AccessPtr - Address == 1) ? "" : "s", Size,
               Address);
    } else if (AccessPtr < Address) {
      snprintf(DescriptionBuffer, kDescriptionBufferLen,
               "(%zu byte%s to the left of a %zu-byte allocation at 0x%zx) ",
               Address - AccessPtr, (Address - AccessPtr == 1) ? "" : "s", Size,
               Address);
    } else if (AccessPtr > Address) {
      snprintf(DescriptionBuffer, kDescriptionBufferLen,
               "(%zu byte%s to the right of a %zu-byte allocation at 0x%zx) ",
               AccessPtr - Address, (AccessPtr - Address == 1) ? "" : "s", Size,
               Address);
    } else {
      snprintf(DescriptionBuffer, kDescriptionBufferLen,
               "(a %zu-byte allocation) ", Size);
    }
  }

  // Possible number of digits of a 64-bit number: ceil(log10(2^64)) == 20. Add
  // a null terminator, and round to the nearest 8-byte boundary.
  uint64_t ThreadID = gwp_asan::getThreadID();
  constexpr size_t kThreadBufferLen = 24;
  char ThreadBuffer[kThreadBufferLen];
  if (ThreadID == gwp_asan::kInvalidThreadID)
    snprintf(ThreadBuffer, kThreadBufferLen, "<unknown>");
  else
    snprintf(ThreadBuffer, kThreadBufferLen, "%" PRIu64, ThreadID);

  Printf("%s at 0x%zx %sby thread %s here:\n", gwp_asan::ErrorToString(E),
         AccessPtr, DescriptionBuffer, ThreadBuffer);
}

void defaultPrintStackTrace(uintptr_t *Trace, size_t TraceLength,
                            gwp_asan::crash_handler::Printf_t Printf) {
  if (TraceLength == 0)
    Printf("  <unknown (does your allocator support backtracing?)>\n");

  for (size_t i = 0; i < TraceLength; ++i) {
    Printf("  #%zu 0x%zx in <unknown>\n", i, Trace[i]);
  }
  Printf("\n");
}

} // anonymous namespace

namespace gwp_asan {
namespace crash_handler {
PrintBacktrace_t getBasicPrintBacktraceFunction() {
  return defaultPrintStackTrace;
}

void installSignalHandlers(gwp_asan::GuardedPoolAllocator *GPA, Printf_t Printf,
                           PrintBacktrace_t PrintBacktrace,
                           SegvBacktrace_t SegvBacktrace) {
  GPAForSignalHandler = GPA;
  PrintfForSignalHandler = Printf;
  PrintBacktraceForSignalHandler = PrintBacktrace;
  BacktraceForSignalHandler = SegvBacktrace;

  struct sigaction Action;
  Action.sa_sigaction = sigSegvHandler;
  Action.sa_flags = SA_SIGINFO;
  sigaction(SIGSEGV, &Action, &PreviousHandler);
  SignalHandlerInstalled = true;
}

void uninstallSignalHandlers() {
  if (SignalHandlerInstalled) {
    sigaction(SIGSEGV, &PreviousHandler, nullptr);
    SignalHandlerInstalled = false;
  }
}

void dumpReport(uintptr_t ErrorPtr, const gwp_asan::AllocatorState *State,
                const gwp_asan::AllocationMetadata *Metadata,
                SegvBacktrace_t SegvBacktrace, Printf_t Printf,
                PrintBacktrace_t PrintBacktrace, void *Context) {
  assert(State && "dumpReport missing Allocator State.");
  assert(Metadata && "dumpReport missing Metadata.");
  assert(Printf && "dumpReport missing Printf.");

  if (!__gwp_asan_error_is_mine(State, ErrorPtr))
    return;

  Printf("*** GWP-ASan detected a memory error ***\n");
  ScopedEndOfReportDecorator Decorator(Printf);

  uintptr_t InternalErrorPtr = __gwp_asan_get_internal_crash_address(State);
  if (InternalErrorPtr != 0u)
    ErrorPtr = InternalErrorPtr;

  Error E = __gwp_asan_diagnose_error(State, Metadata, ErrorPtr);

  if (E == Error::UNKNOWN) {
    Printf("GWP-ASan cannot provide any more information about this error. "
           "This may occur due to a wild memory access into the GWP-ASan pool, "
           "or an overflow/underflow that is > 512B in length.\n");
    return;
  }

  const gwp_asan::AllocationMetadata *AllocMeta =
      __gwp_asan_get_metadata(State, Metadata, ErrorPtr);

  // Print the error header.
  printHeader(E, ErrorPtr, AllocMeta, Printf);

  // Print the fault backtrace.
  static constexpr unsigned kMaximumStackFramesForCrashTrace = 512;
  uintptr_t Trace[kMaximumStackFramesForCrashTrace];
  size_t TraceLength =
      SegvBacktrace(Trace, kMaximumStackFramesForCrashTrace, Context);

  PrintBacktrace(Trace, TraceLength, Printf);

  if (AllocMeta == nullptr)
    return;

  // Maybe print the deallocation trace.
  if (__gwp_asan_is_deallocated(AllocMeta)) {
    uint64_t ThreadID = __gwp_asan_get_deallocation_thread_id(AllocMeta);
    if (ThreadID == kInvalidThreadID)
      Printf("0x%zx was deallocated by thread <unknown> here:\n", ErrorPtr);
    else
      Printf("0x%zx was deallocated by thread %zu here:\n", ErrorPtr, ThreadID);
    TraceLength = __gwp_asan_get_deallocation_trace(
        AllocMeta, Trace, kMaximumStackFramesForCrashTrace);
    PrintBacktrace(Trace, TraceLength, Printf);
  }

  // Print the allocation trace.
  uint64_t ThreadID = __gwp_asan_get_allocation_thread_id(AllocMeta);
  if (ThreadID == kInvalidThreadID)
    Printf("0x%zx was allocated by thread <unknown> here:\n", ErrorPtr);
  else
    Printf("0x%zx was allocated by thread %zu here:\n", ErrorPtr, ThreadID);
  TraceLength = __gwp_asan_get_allocation_trace(
      AllocMeta, Trace, kMaximumStackFramesForCrashTrace);
  PrintBacktrace(Trace, TraceLength, Printf);
}
} // namespace crash_handler
} // namespace gwp_asan
