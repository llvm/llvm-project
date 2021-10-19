// RUN: %clang_cc1 -triple riscv32 -O1 -emit-llvm %s -o - \
// RUN:   | FileCheck %s -check-prefix=RV32I
// RUN: %clang_cc1 -triple riscv32 -target-feature +a -O1 -emit-llvm %s -o - \
// RUN:   | FileCheck %s -check-prefix=RV32IA
// RUN: %clang_cc1 -triple riscv64 -O1 -emit-llvm %s -o - \
// RUN:   | FileCheck %s -check-prefix=RV64I
// RUN: %clang_cc1 -triple riscv64 -target-feature +a -O1 -emit-llvm %s -o - \
// RUN:   | FileCheck %s -check-prefix=RV64IA

// This test demonstrates that MaxAtomicInlineWidth is set appropriately when
// the atomics instruction set extension is enabled.

#include <stdatomic.h>
#include <stdint.h>

void test_i8_atomics(_Atomic(int8_t) * a, int8_t b) {
  // RV32I:  call zeroext i8 @__atomic_load_1
  // RV32I:  call void @__atomic_store_1
  // RV32I:  call zeroext i8 @__atomic_fetch_add_1
  // RV32IA: load atomic i8, i8* %a seq_cst, align 1
  // RV32IA: store atomic i8 %b, i8* %a seq_cst, align 1
  // RV32IA: atomicrmw add i8* %a, i8 %b seq_cst, align 1
  // RV64I:  call zeroext i8 @__atomic_load_1
  // RV64I:  call void @__atomic_store_1
  // RV64I:  call zeroext i8 @__atomic_fetch_add_1
  // RV64IA: load atomic i8, i8* %a seq_cst, align 1
  // RV64IA: store atomic i8 %b, i8* %a seq_cst, align 1
  // RV64IA: atomicrmw add i8* %a, i8 %b seq_cst, align 1
  __c11_atomic_load(a, memory_order_seq_cst);
  __c11_atomic_store(a, b, memory_order_seq_cst);
  __c11_atomic_fetch_add(a, b, memory_order_seq_cst);
}

void test_i32_atomics(_Atomic(int32_t) * a, int32_t b) {
  // RV32I:  call i32 @__atomic_load_4
  // RV32I:  call void @__atomic_store_4
  // RV32I:  call i32 @__atomic_fetch_add_4
  // RV32IA: load atomic i32, i32* %a seq_cst, align 4
  // RV32IA: store atomic i32 %b, i32* %a seq_cst, align 4
  // RV32IA: atomicrmw add i32* %a, i32 %b seq_cst, align 4
  // RV64I:  call signext i32 @__atomic_load_4
  // RV64I:  call void @__atomic_store_4
  // RV64I:  call signext i32 @__atomic_fetch_add_4
  // RV64IA: load atomic i32, i32* %a seq_cst, align 4
  // RV64IA: store atomic i32 %b, i32* %a seq_cst, align 4
  // RV64IA: atomicrmw add i32* %a, i32 %b seq_cst, align 4
  __c11_atomic_load(a, memory_order_seq_cst);
  __c11_atomic_store(a, b, memory_order_seq_cst);
  __c11_atomic_fetch_add(a, b, memory_order_seq_cst);
}

void test_i64_atomics(_Atomic(int64_t) * a, int64_t b) {
  // RV32I:  call i64 @__atomic_load_8
  // RV32I:  call void @__atomic_store_8
  // RV32I:  call i64 @__atomic_fetch_add_8
  // RV32IA: call i64 @__atomic_load_8
  // RV32IA: call void @__atomic_store_8
  // RV32IA: call i64 @__atomic_fetch_add_8
  // RV64I:  call i64 @__atomic_load_8
  // RV64I:  call void @__atomic_store_8
  // RV64I:  call i64 @__atomic_fetch_add_8
  // RV64IA: load atomic i64, i64* %a seq_cst, align 8
  // RV64IA: store atomic i64 %b, i64* %a seq_cst, align 8
  // RV64IA: atomicrmw add i64* %a, i64 %b seq_cst, align 8
  __c11_atomic_load(a, memory_order_seq_cst);
  __c11_atomic_store(a, b, memory_order_seq_cst);
  __c11_atomic_fetch_add(a, b, memory_order_seq_cst);
}
