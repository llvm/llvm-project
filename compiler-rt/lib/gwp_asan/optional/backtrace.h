//===-- backtrace.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GWP_ASAN_OPTIONAL_BACKTRACE_H_
#define GWP_ASAN_OPTIONAL_BACKTRACE_H_

#include "gwp_asan/options.h"

namespace gwp_asan {
namespace options {
// Functions to get the platform-specific and implementation-specific backtrace
// and backtrace printing functions when RTGwpAsanBacktraceLibc or
// RTGwpAsanBacktraceSanitizerCommon are linked. Use these functions to get the
// backtrace function for populating the Options::Backtrace and
// Options::PrintBacktrace when initialising the GuardedPoolAllocator. Please
// note any thread-safety descriptions for the implementation of these functions
// that you use.
Backtrace_t getBacktraceFunction();
PrintBacktrace_t getPrintBacktraceFunction();
} // namespace options
} // namespace gwp_asan

#endif // GWP_ASAN_OPTIONAL_BACKTRACE_H_
