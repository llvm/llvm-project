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
#include "rtsan/rtsan_flags.h"

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"

using namespace __sanitizer;
using namespace __rtsan;

static atomic_uint32_t total_error_count{0};
static atomic_uint32_t unique_error_count{0};
static atomic_uint32_t suppressed_count{0};

void __rtsan::IncrementTotalErrorCount() {
  atomic_fetch_add(&total_error_count, 1, memory_order_relaxed);
}

void __rtsan::IncrementUniqueErrorCount() {
  atomic_fetch_add(&unique_error_count, 1, memory_order_relaxed);
}

static u32 GetTotalErrorCount() {
  return atomic_load(&total_error_count, memory_order_relaxed);
}

static u32 GetUniqueErrorCount() {
  return atomic_load(&unique_error_count, memory_order_relaxed);
}

void __rtsan::IncrementSuppressedCount() {
  atomic_fetch_add(&suppressed_count, 1, memory_order_relaxed);
}

static u32 GetSuppressedCount() {
  return atomic_load(&suppressed_count, memory_order_relaxed);
}

void __rtsan::PrintStatisticsSummary() {
  ScopedErrorReportLock l;
  Printf("RealtimeSanitizer exit stats:\n");
  Printf("    Total error count: %u\n", GetTotalErrorCount());
  Printf("    Unique error count: %u\n", GetUniqueErrorCount());

  if (flags().ContainsSuppresionFile())
    Printf("    Suppression count: %u\n", GetSuppressedCount());
}
