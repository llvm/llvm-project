//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: clang
// UNSUPPORTED: libcpp-has-no-threads

// Adding "-fsanitize=thread" directly causes many platforms to fail (because
// they don't support tsan), and causes other sanitizer builds to fail (e.g.
// asan and tsan don't mix). Instead, require the tsan feature.
// REQUIRES: tsan

// This test verifies behavior specified by [atomics.types.operations.req]/21:
//
//     When only one memory_order argument is supplied, the value of success is
//     order, and the value of failure is order except that a value of
//     memory_order_acq_rel shall be replaced by the value memory_order_acquire
//     and a value of memory_order_release shall be replaced by the value
//     memory_order_relaxed.
//
// This test mirrors replace_failure_order.pass.cpp. However, we also want to
// verify the codegen is correct. This verifies a bug where memory_order_acq_rel
// was not being replaced with memory_order_acquire in external
// TSAN-instrumented tests.

// RUN: %{cxx} -c %s %{flags} %{compile_flags} -O2 -stdlib=libc++ -S -emit-llvm -o %t.ll

#include <atomic>

// Note: libc++ tests do not use on FileCheck.
// RUN: grep -E "call i32 @__tsan_atomic32_compare_exchange_val\(.*, i32 1, i32 4, i32 2\)" %t.ll
bool strong_memory_order_acq_rel(std::atomic<int>* a, int cmp) {
  return a->compare_exchange_strong(cmp, 1, std::memory_order_acq_rel);
}
