//===-- segv_handler_fuchsia.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gwp_asan/optional/segv_handler.h"

// GWP-ASan on Fuchsia doesn't currently support signal handlers.

namespace gwp_asan {
namespace crash_handler {
void installSignalHandlers(gwp_asan::GuardedPoolAllocator * /* GPA */,
                           Printf_t /* Printf */,
                           PrintBacktrace_t /* PrintBacktrace */,
                           SegvBacktrace_t /* SegvBacktrace */) {}

void uninstallSignalHandlers() {}
} // namespace crash_handler
} // namespace gwp_asan
