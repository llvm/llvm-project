//===--- rtsan_stats.cpp - Realtime Sanitizer -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of the RealtimeSanitizer runtime library
//
//===----------------------------------------------------------------------===//

#include "rtsan/rtsan_stats.h"

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"

using namespace __sanitizer;
using namespace __rtsan;

static atomic_uint32_t rtsan_total_error_count{0};

void __rtsan::IncrementTotalErrorCount() {
  atomic_fetch_add(&rtsan_total_error_count, 1, memory_order_relaxed);
}

static u32 GetTotalErrorCount() {
  return atomic_load(&rtsan_total_error_count, memory_order_relaxed);
}

void __rtsan::PrintStatisticsSummary() {
  ScopedErrorReportLock l;
  Printf("RealtimeSanitizer exit stats:\n");
  Printf("    Total error count: %u\n", GetTotalErrorCount());
}
