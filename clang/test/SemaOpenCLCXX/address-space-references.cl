// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=clc++ -pedantic -verify -fsyntax-only

__global const int& f(__global float &ref) {
  return ref; // expected-error{{reference of type 'const __global int &' cannot bind to a temporary object because of address space mismatch}}
}

int bar(const __global unsigned int &i); // expected-note{{passing argument to parameter 'i' here}}
//FIXME: With the overload below the call should be resolved
// successfully. However, current overload resolution logic
// can't detect this case and therefore fails.
int bar(const unsigned int &i);

void foo() {
  bar(1); // expected-error{{binding reference of type 'const __global unsigned int' to value of type 'int' changes address space}}
}

// Test addr space conversion with nested pointers

extern void nestptr(int *&); // expected-note {{candidate function not viable: no known conversion from '__global int *__private' to '__generic int *__generic &__private' for 1st argument}}
extern void nestptr_const(int * const &); // expected-note {{candidate function not viable: cannot pass pointer to address space '__constant' as a pointer to address space '__generic' in 1st argument}}
int test_nestptr(__global int *glob, __constant int *cons, int* gen) {
  nestptr(glob); // expected-error{{no matching function for call to 'nestptr'}}
  // Addr space conversion first occurs on a temporary.
  nestptr_const(glob);
  // No legal conversion between disjoint addr spaces.
  nestptr_const(cons); // expected-error{{no matching function for call to 'nestptr_const'}}
  return *(*cons ? glob : gen);
}
