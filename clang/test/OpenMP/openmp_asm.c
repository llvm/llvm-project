// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify=fopenmp -emit-llvm -o - %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -verify -emit-llvm -o - %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -x c++ -fopenmp -verify=fopenmp -emit-llvm -o - %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -x c++ -verify -emit-llvm -o - %s

// fopenmp-no-diagnostics

void t1(int *a, int *b) {
  asm volatile("" : "+&r"(a) : ""(b)); // expected-error {{invalid input constraint '' in asm}}
}

void t2() {
  asm ("nop" : : : "foo"); // expected-error {{unknown register name 'foo' in asm}}
}

void t3() {
  asm goto ("" ::: "unwind" : label); // expected-error {{unwind clobber cannot be used with asm goto}}
label:
  ;
}

typedef int vec256 __attribute__((ext_vector_type(8)));
vec256 t4() {
  vec256 out;
  asm("something %0" : "=y"(out)); // expected-error {{invalid output size for constraint '=y'}}
  return out;
}
