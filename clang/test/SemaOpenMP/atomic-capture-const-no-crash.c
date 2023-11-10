// RUN: %clang_cc1 -fopenmp-simd -fsyntax-only -verify %s
// see https://github.com/llvm/llvm-project/issues/69069 
// or https://github.com/llvm/llvm-project/pull/71480

void test() {
  int v; const int x; // expected-note {{variable 'x' declared const here}}
#pragma omp atomic capture
  { 
    v = x; 
    x = 1; // expected-error {{cannot assign to variable 'x' with const-qualified type 'const int'}} 
  }
}