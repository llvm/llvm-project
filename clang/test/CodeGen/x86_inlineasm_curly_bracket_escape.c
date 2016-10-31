// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -O0  -S -emit-llvm -o - -Wall -Werror | FileCheck %s
// This test checks validity of inline assembly using curly brackets syntax
// for extended inline asm.

void test_curly_brackets() {
    //CHECK:  %xmm1,%xmm0,%xmm1 {%k1}{z}
    asm("vpaddb\t %%xmm1,%%xmm0,%%xmm1 %{%%k1%}%{z%}\t":::);
}