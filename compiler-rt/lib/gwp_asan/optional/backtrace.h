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
// and backtrace printing functions.
Backtrace_t getBacktraceFunction();
PrintBacktrace_t getPrintBacktraceFunction();
} // namespace options
} // namespace gwp_asan

#endif // GWP_ASAN_OPTIONAL_BACKTRACE_H_
