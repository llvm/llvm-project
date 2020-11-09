//===-- backtrace_fuchsia.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gwp_asan/optional/backtrace.h"

// GWP-ASan on Fuchsia doesn't currently support backtraces.

namespace gwp_asan {
namespace options {
Backtrace_t getBacktraceFunction() { return nullptr; }
crash_handler::PrintBacktrace_t getPrintBacktraceFunction() { return nullptr; }
} // namespace options

namespace crash_handler {
SegvBacktrace_t getSegvBacktraceFunction() { return nullptr; }
} // namespace crash_handler
} // namespace gwp_asan
