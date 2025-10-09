// RUN: %clang_cc1 -triple riscv32 -O1 -emit-llvm %s -o - \
// RUN:   -verify=no-atomics
// RUN: %clang_cc1 -triple riscv32 -target-feature +a -O1 -emit-llvm %s -o - \
// RUN:   -verify=small-atomics
// RUN: %clang_cc1 -triple riscv64 -O1 -emit-llvm %s -o - \
// RUN:   -verify=no-atomics
// RUN: %clang_cc1 -triple riscv64 -target-feature +a -O1 -emit-llvm %s -o - \
// RUN:   -verify=all-atomics

// all-atomics-no-diagnostics

#include <stdatomic.h>
#include <stdint.h>

void test_i8_atomics(_Atomic(int8_t) * a, int8_t b) {
  __c11_atomic_load(a, memory_order_seq_cst);         // no-atomics-warning {{large atomic operation may incur significant performance penalty; the access size (1 bytes) exceeds the max lock-free size (0 bytes)}}
  __c11_atomic_store(a, b, memory_order_seq_cst);     // no-atomics-warning {{large atomic operation may incur significant performance penalty; the access size (1 bytes) exceeds the max lock-free size (0 bytes)}}
  __c11_atomic_fetch_add(a, b, memory_order_seq_cst); // no-atomics-warning {{large atomic operation may incur significant performance penalty; the access size (1 bytes) exceeds the max lock-free size (0 bytes)}}
}

void test_i32_atomics(_Atomic(int32_t) * a, int32_t b) {
  __c11_atomic_load(a, memory_order_seq_cst);         // no-atomics-warning {{large atomic operation may incur significant performance penalty; the access size (4 bytes) exceeds the max lock-free size (0 bytes)}}
  __c11_atomic_store(a, b, memory_order_seq_cst);     // no-atomics-warning {{large atomic operation may incur significant performance penalty; the access size (4 bytes) exceeds the max lock-free size (0 bytes)}}
  __c11_atomic_fetch_add(a, b, memory_order_seq_cst); // no-atomics-warning {{large atomic operation may incur significant performance penalty; the access size (4 bytes) exceeds the max lock-free size (0 bytes)}}
}

void test_i64_atomics(_Atomic(int64_t) * a, int64_t b) {
  __c11_atomic_load(a, memory_order_seq_cst);         // no-atomics-warning {{large atomic operation may incur significant performance penalty; the access size (8 bytes) exceeds the max lock-free size (0 bytes)}}
                                                      // small-atomics-warning@28 {{large atomic operation may incur significant performance penalty; the access size (8 bytes) exceeds the max lock-free size (4 bytes)}}
  __c11_atomic_store(a, b, memory_order_seq_cst);     // no-atomics-warning {{large atomic operation may incur significant performance penalty; the access size (8 bytes) exceeds the max lock-free size (0 bytes)}}
                                                      // small-atomics-warning@30 {{large atomic operation may incur significant performance penalty; the access size (8 bytes) exceeds the max lock-free size (4 bytes)}}
  __c11_atomic_fetch_add(a, b, memory_order_seq_cst); // no-atomics-warning {{large atomic operation may incur significant performance penalty; the access size (8 bytes) exceeds the max lock-free size (0 bytes)}}
                                                      // small-atomics-warning@32 {{large atomic operation may incur significant performance penalty; the access size (8 bytes) exceeds the max lock-free size (4 bytes)}}
}
