// RUN: %clang_cc1 -verify -triple x86_64-unknown-unknown -emit-llvm-only %s
// RUN: %clang_cc1 -verify -triple x86_64-unknown-unknown -emit-llvm-only -fopenmp %s
// RUN: %clang_cc1 -verify -triple x86_64-unknown-unknown -emit-llvm-only -fopenmp -fopenmp-version=45 %s
// RUN: %clang_cc1 -verify -triple x86_64-unknown-unknown -emit-llvm-only -fopenmp-simd %s

// An invalid asm constraint must be diagnosed rather than reaching CodeGen.

void f(void) {
  __asm__("nop" ::: "no_such_register"); // expected-error {{unknown register name 'no_such_register' in asm}}
}

// Same when CodeGen would emit the function before the end of the TU: always
// under OpenMP 4.5, and under 5.x once a later 'declare target' names it.
void g(int x) {
  __asm__("nop" : "=@"(x)); // expected-error {{invalid output constraint '=@' in asm}}
}
#pragma omp declare target to(g)
