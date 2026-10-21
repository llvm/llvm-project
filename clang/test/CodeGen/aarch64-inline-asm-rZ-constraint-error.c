// RUN: not %clang_cc1 -triple aarch64-linux-gnu -O2 -S -o /dev/null %s 2>&1 | FileCheck %s

// Test that the "rZ" inline assembly constraint properly rejects non-constant values.
// The "rZ" constraint is only valid for literal zero values.

// CHECK: error: invalid operand for inline asm constraint 'rZ'
void test_rZ_runtime_value(long *addr, long val) {
    __asm__ volatile("str %1, [%0]" : : "r"(addr), "rZ"(val));
}

// CHECK: error: invalid operand for inline asm constraint 'rZ'
void test_rZ_runtime_i32(int *addr, int val) {
    __asm__ volatile("str %w1, [%0]" : : "r"(addr), "rZ"(val));
}

// CHECK: error: invalid operand for inline asm constraint 'rZ'
void test_rZ_non_constant(long val) {
    __asm__ volatile("mov x2, %0" : : "rZ"(val));
}
