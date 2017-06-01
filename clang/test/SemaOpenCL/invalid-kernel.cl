// RUN: %clang_cc1 -verify %s
// RUN: %clang_cc1 -cl-std=CL2.0 -verify %s

kernel void no_ptrptr(global int * global *i) { } // expected-error{{kernel parameter cannot be declared as a pointer to a pointer}}

__kernel void no_privateptr(__private int *i) { } // expected-error {{pointer arguments to kernel functions must reside in '__global', '__constant' or '__local' address space}}

__kernel void no_privatearray(__private int i[]) { } // expected-error {{pointer arguments to kernel functions must reside in '__global', '__constant' or '__local' address space}}

kernel int bar()  { // expected-error {{kernel must have void return type}}
  return 6;
}

kernel void main() { // expected-error {{kernel cannot be called 'main'}}

}

int main() { // expected-error {{function cannot be called 'main'}}
  return 0;
}

int* global x(int* x) { // expected-error {{return value cannot be qualified with address space}}
  return x + 1;
}

int* local x(int* x) { // expected-error {{return value cannot be qualified with address space}}
  return x + 1;
}

int* constant x(int* x) { // expected-error {{return value cannot be qualified with address space}}
  return x + 1;
}

__kernel void testKernel(int *ptr) { // expected-error {{pointer arguments to kernel functions must reside in '__global', '__constant' or '__local' address space}}
}
