// RUN: %clang_cc1 -triple loongarch32 -O1 -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=LA32
// RUN: %clang_cc1 -triple loongarch64 -O1 -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=LA64

/// This test demonstrates that MaxAtomicInlineWidth is set appropriately.

#include <stdatomic.h>
#include <stdint.h>

void test_i8_atomics(_Atomic(int8_t) * a, int8_t b) {
  // LA32: load atomic i8, ptr %a seq_cst, align 1
  // LA32: store atomic i8 %b, ptr %a seq_cst, align 1
  // LA32: atomicrmw add ptr %a, i8 %b seq_cst
  // LA64: load atomic i8, ptr %a seq_cst, align 1
  // LA64: store atomic i8 %b, ptr %a seq_cst, align 1
  // LA64: atomicrmw add ptr %a, i8 %b seq_cst
  __c11_atomic_load(a, memory_order_seq_cst);
  __c11_atomic_store(a, b, memory_order_seq_cst);
  __c11_atomic_fetch_add(a, b, memory_order_seq_cst);
}

void test_i32_atomics(_Atomic(int32_t) * a, int32_t b) {
  // LA32: load atomic i32, ptr %a seq_cst, align 4
  // LA32: store atomic i32 %b, ptr %a seq_cst, align 4
  // LA32: atomicrmw add ptr %a, i32 %b seq_cst
  // LA64: load atomic i32, ptr %a seq_cst, align 4
  // LA64: store atomic i32 %b, ptr %a seq_cst, align 4
  // LA64: atomicrmw add ptr %a, i32 %b seq_cst
  __c11_atomic_load(a, memory_order_seq_cst);
  __c11_atomic_store(a, b, memory_order_seq_cst);
  __c11_atomic_fetch_add(a, b, memory_order_seq_cst);
}

void test_i64_atomics(_Atomic(int64_t) * a, int64_t b) {
  // LA32: call i64 @__atomic_load_8
  // LA32: call void @__atomic_store_8
  // LA32: call i64 @__atomic_fetch_add_8
  // LA64: load atomic i64, ptr %a seq_cst, align 8
  // LA64: store atomic i64 %b, ptr %a seq_cst, align 8
  // LA64: atomicrmw add ptr %a, i64 %b seq_cst
  __c11_atomic_load(a, memory_order_seq_cst);
  __c11_atomic_store(a, b, memory_order_seq_cst);
  __c11_atomic_fetch_add(a, b, memory_order_seq_cst);
}
