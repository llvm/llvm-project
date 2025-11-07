// RUN: %clang_cc1 -verify -triple x86_64-unknown-unknown -emit-llvm-only %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm-only -fopenmp %s
typedef int vec256 __attribute__((ext_vector_type(8)));

vec256 foo(vec256 in) {
  vec256 out;

  asm("something %0" : : "y"(in)); // expected-error-re {{invalid {{.*}} constraint 'y'}}
  asm("something %0" : "=y"(out)); // expected-error-re {{invalid {{.*}} constraint '=y'}} omp-error-re {{invalid {{.*}} constraint 'y'}}
  asm("something %0, %0" : "+y"(out)); // expected-error-re {{invalid {{.*}}}} omp-error-re {{invalid {{.*}}}}

  return out;
}
