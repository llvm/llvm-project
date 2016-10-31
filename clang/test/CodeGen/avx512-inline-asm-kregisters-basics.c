// RUN: %clang_cc1 %s -target-cpu skylake-avx512 -O0  -S -o - -Wall -Werror | FileCheck %s
// This test checks basic inline assembly recognition of k0-k7 registers for avx512.

void test_basic_inline_asm_with_k_regs() {
    //CHECK: #APP
    //CHECK: kandw %k1, %k2, %k3
    //CHECK: #NO_APP
    asm("kandw %k1, %k2, %k3\t");
    //CHECK: #APP
    //CHECK: kandw %k4, %k5, %k6
    //CHECK: #NO_APP
    asm("kandw %k4, %k5, %k6\t");
    //CHECK: #APP
    //CHECK: kandw %k7, %k0, %k1
    //CHECK: #NO_APP
    asm("kandw %k7, %k0, %k1\t");
}