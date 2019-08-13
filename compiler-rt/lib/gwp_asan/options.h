//===-- options.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GWP_ASAN_OPTIONS_H_
#define GWP_ASAN_OPTIONS_H_

#include <stddef.h>
#include <stdint.h>

namespace gwp_asan {
namespace options {
// ================================ Requirements ===============================
// This function is required to be implemented by the supporting allocator. The
// sanitizer::Printf() function can be simply used here.
// ================================ Description ================================
// This function shall produce output according to a strict subset of the C
// standard library's printf() family. This function must support printing the
// following formats:
//   1. integers: "%([0-9]*)?(z|ll)?{d,u,x,X}"
//   2. pointers: "%p"
//   3. strings:  "%[-]([0-9]*)?(\\.\\*)?s"
//   4. chars:    "%c"
// =================================== Notes ===================================
// This function has a slightly different signature than the C standard
// library's printf(). Notably, it returns 'void' rather than 'int'.
typedef void (*Printf_t)(const char *Format, ...);

// ================================ Requirements ===============================
// This function is required to be either implemented by the supporting
// allocator, or one of the two provided implementations may be used
// (RTGwpAsanBacktraceLibc or RTGwpAsanBacktraceSanitizerCommon).
// ================================ Description ================================
// This function shall collect the backtrace for the calling thread and place
// the result in `TraceBuffer`. This function should elide itself and all frames
// below itself from `TraceBuffer`, i.e. the caller's frame should be in
// TraceBuffer[0], and subsequent frames 1..n into TraceBuffer[1..n], where a
// maximum of `Size` frames are stored. Returns the number of frames stored into
// `TraceBuffer`, and zero on failure. If the return value of this function is
// equal to `Size`, it may indicate that the backtrace is truncated.
// =================================== Notes ===================================
// This function may directly or indirectly call malloc(), as the
// GuardedPoolAllocator contains a reentrancy barrier to prevent infinite
// recursion. Any allocation made inside this function will be served by the
// supporting allocator, and will not have GWP-ASan protections.
typedef size_t (*Backtrace_t)(uintptr_t *TraceBuffer, size_t Size);

// ================================ Requirements ===============================
// This function is optional for the supporting allocator, but one of the two
// provided implementations may be used (RTGwpAsanBacktraceLibc or
// RTGwpAsanBacktraceSanitizerCommon). If not provided, a default implementation
// is used which prints the raw pointers only.
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

struct Options {
  Printf_t Printf = nullptr;
  Backtrace_t Backtrace = nullptr;
  PrintBacktrace_t PrintBacktrace = nullptr;

  // Read the options from the included definitions file.
#define GWP_ASAN_OPTION(Type, Name, DefaultValue, Description)                 \
  Type Name = DefaultValue;
#include "gwp_asan/options.inc"
#undef GWP_ASAN_OPTION

  void setDefaults() {
#define GWP_ASAN_OPTION(Type, Name, DefaultValue, Description)                 \
  Name = DefaultValue;
#include "gwp_asan/options.inc"
#undef GWP_ASAN_OPTION

    Printf = nullptr;
    Backtrace = nullptr;
    PrintBacktrace = nullptr;
  }
};
} // namespace options
} // namespace gwp_asan

#endif // GWP_ASAN_OPTIONS_H_
