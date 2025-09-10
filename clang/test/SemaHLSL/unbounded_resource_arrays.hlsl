// RUN: not %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -finclude-default-header --o - %s -verify

// unbounded resource array at a global scope
RWBuffer<float> unbounded_array[]; // no_error

// expected-error@+1 {{incomplete resource array in a function parameter}}
void foo(RWBuffer<float> array_arg[]) {}

RWBuffer<float> A, B;

[numthreads(4,1,1)]
void main() {
  // expected-error@+1{{definition of variable with array type needs an explicit size or an initializer}}
  RWBuffer<float> res_local_array1[]; 

  // expected-error@+1{{array initializer must be an initialzer list}}
  RWBuffer<float> res_local_array2[] = unbounded_array;

  // local incomplete resource array with initializer
  RWBuffer<float> res_local_array3[] = { A, B }; // no error
}
