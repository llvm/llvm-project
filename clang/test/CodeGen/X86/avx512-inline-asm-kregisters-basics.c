// RUN: %clang_cc1 %s -O0 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -emit-llvm -o - -Wall -Werror | FileCheck %s
// This test checks basic inline assembly recognition of k0-k7 registers for avx512.

void test_basic_inline_asm_with_k_regs(void) {
    //CHECK: kandw %k1, %k2, %k3
    asm("kandw %k1, %k2, %k3\t");
    //CHECK: kandw %k4, %k5, %k6
    asm("kandw %k4, %k5, %k6\t");
    //CHECK: kandw %k7, %k0, %k1
    asm("kandw %k7, %k0, %k1\t");
}