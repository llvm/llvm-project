// RUN: %clang_cc1 -verify -triple x86_64-unknown-unknown -emit-llvm-only %s
// RUN: %clang_cc1 -verify -triple x86_64-unknown-unknown -emit-llvm-only -fopenmp %s
// RUN: %clang_cc1 -verify -triple x86_64-unknown-unknown -emit-llvm-only -fopenmp-simd %s

// An invalid asm constraint must be diagnosed rather than reaching CodeGen.

void f(void) {
  __asm__("nop" ::: "no_such_register"); // expected-error {{unknown register name 'no_such_register' in asm}}
}
