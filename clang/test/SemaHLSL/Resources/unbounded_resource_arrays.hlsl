// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -x hlsl -fsyntax-only -finclude-default-header %s -verify

// unbounded resource array parameter whose resource type has not been used earlier
// expected-error@+1 {{incomplete resource array in a function parameter}}
void no_prior_use(RWByteAddressBuffer array_arg[]) {}

// unbounded resource array at a global scope
RWBuffer<float> unbounded_array[]; // no error

// expected-error@+1 {{incomplete resource array in a function parameter}}
void foo(RWBuffer<float> array_arg[]) {}

// a non-resource incomplete-array parameter should not error
struct S;
void non_resource(S array_arg[]); // no error

RWBuffer<float> A, B;

[numthreads(4,1,1)]
void main() {
  // expected-error@+1{{definition of variable with array type needs an explicit size or an initializer}}
  RWBuffer<float> res_local_array1[]; 

  // expected-error@+1{{array initializer must be an initializer list}}
  RWBuffer<float> res_local_array2[] = unbounded_array;

  // local incomplete resource array with initializer
  RWBuffer<float> res_local_array3[] = { A, B }; // no error
}
