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
// The function pointer type for printf(). Follows the standard format from the
// sanitizers library. If the supported allocator exposes printing via a
// different function signature, please provide a wrapper which has this
// printf() signature, and pass the wrapper instead.
typedef void (*Printf_t)(const char *Format, ...);

// The function pointer type for backtrace information. Required to be
// implemented by the supporting allocator. The callee should elide itself and
// all frames below itself from TraceBuffer, i.e. the caller's frame should be
// in TraceBuffer[0], and subsequent frames 1..n into TraceBuffer[1..n], where a
// maximum of `MaximumDepth - 1` frames are stored. TraceBuffer should be
// nullptr-terminated (i.e. if there are 5 frames; TraceBuffer[5] == nullptr).
// If the allocator cannot supply backtrace information, it should set
// TraceBuffer[0] == nullptr.
typedef void (*Backtrace_t)(uintptr_t *TraceBuffer, size_t Size);
typedef void (*PrintBacktrace_t)(uintptr_t *TraceBuffer, Printf_t Print);

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
