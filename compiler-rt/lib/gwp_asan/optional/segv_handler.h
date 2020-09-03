//===-- crash_handler.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GWP_ASAN_OPTIONAL_CRASH_HANDLER_H_
#define GWP_ASAN_OPTIONAL_CRASH_HANDLER_H_

#include "gwp_asan/guarded_pool_allocator.h"
#include "gwp_asan/options.h"

namespace gwp_asan {
namespace crash_handler {
// ================================ Requirements ===============================
// This function must be provided by the supporting allocator only when this
// provided crash handler is used to dump the generic report.
// sanitizer::Printf() function can be simply used here.
// ================================ Description ================================
// This function shall produce output according to a strict subset of the C
// standard library's printf() family. This function must support printing the
// following formats:
//   1. integers: "%([0-9]*)?(z|ll)?{d,u,x,X}"
//   2. pointers: "%p"
//   3. strings:  "%[-]([0-9]*)?(\\.\\*)?s"
//   4. chars:    "%c"
// This function must be implemented in a signal-safe manner, and thus must not
// malloc().
// =================================== Notes ===================================
// This function has a slightly different signature than the C standard
// library's printf(). Notably, it returns 'void' rather than 'int'.
typedef void (*Printf_t)(const char *Format, ...);

// ================================ Requirements ===============================
// This function is required for the supporting allocator, but one of the three
// provided implementations may be used (RTGwpAsanBacktraceLibc,
// RTGwpAsanBacktraceSanitizerCommon, or BasicPrintBacktraceFunction).
// ================================ Description ================================
// This function shall take the backtrace provided in `TraceBuffer`, and print
// it in a human-readable format using `Print`. Generally, this function shall
// resolve raw pointers to section offsets and print them with the following
// sanitizer-common format:
//      "  #{frame_number} {pointer} in {function name} ({binary name}+{offset}"
// e.g. "  #5 0x420459 in _start (/tmp/uaf+0x420459)"
// This format allows the backtrace to be symbolized offline successfully using
// llvm-symbolizer.
// =================================== Notes ===================================
// This function may directly or indirectly call malloc(), as the
// GuardedPoolAllocator contains a reentrancy barrier to prevent infinite
// recursion. Any allocation made inside this function will be served by the
// supporting allocator, and will not have GWP-ASan protections.
typedef void (*PrintBacktrace_t)(uintptr_t *TraceBuffer, size_t TraceLength,
                                 Printf_t Print);

// Returns a function pointer to a basic PrintBacktrace implementation. This
// implementation simply prints the stack trace in a human readable fashion
// without any symbolization.
PrintBacktrace_t getBasicPrintBacktraceFunction();

// Install the SIGSEGV crash handler for printing use-after-free and heap-
// buffer-{under|over}flow exceptions if the user asked for it. This is platform
// specific as even though POSIX and Windows both support registering handlers
// through signal(), we have to use platform-specific signal handlers to obtain
// the address that caused the SIGSEGV exception. GPA->init() must be called
// before this function.
void installSignalHandlers(gwp_asan::GuardedPoolAllocator *GPA, Printf_t Printf,
                           PrintBacktrace_t PrintBacktrace,
                           options::Backtrace_t Backtrace);

void uninstallSignalHandlers();

void dumpReport(uintptr_t ErrorPtr, const gwp_asan::AllocatorState *State,
                const gwp_asan::AllocationMetadata *Metadata,
                options::Backtrace_t Backtrace, Printf_t Printf,
                PrintBacktrace_t PrintBacktrace);
} // namespace crash_handler
} // namespace gwp_asan

#endif // GWP_ASAN_OPTIONAL_CRASH_HANDLER_H_
