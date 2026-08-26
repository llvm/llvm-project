// RUN: %clang_cc1 -verify -triple x86_64-unknown-unknown -emit-llvm-only %s
// RUN: %clang_cc1 -verify=omp -triple x86_64-unknown-unknown -emit-llvm-only -fopenmp %s
typedef int vec256 __attribute__((ext_vector_type(8)));

// omp-warning@+2 {{AVX vector return of type 'vec256' (vector of 8 'int' values) without 'avx' enabled changes the ABI}}
// omp-warning@+1 {{AVX vector argument of type 'vec256' (vector of 8 'int' values) without 'avx' enabled changes the ABI}}
vec256 foo(vec256 in) {
  vec256 out;

  asm("something %0" : : "y"(in)); // expected-error {{invalid input size for constraint 'y'}}
  // omp-error@+1 {{invalid type 'vec256' (vector of 8 'int' values) in asm input for constraint 'y'}}
  asm("something %0" : "=y"(out)); // expected-error {{invalid output size for constraint '=y'}}
  // omp-error@+1 {{invalid type 'vec256' (vector of 8 'int' values) in asm input for constraint 'y'}}
  asm("something %0, %0" : "+y"(out)); // expected-error {{invalid output size for constraint '+y'}}

  return out;
}
