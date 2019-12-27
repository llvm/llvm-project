// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-std=CL1.2

typedef float float2 __attribute__((ext_vector_type(2)));
typedef long long2 __attribute__((ext_vector_type(2)));
typedef int int2 __attribute__((ext_vector_type(2)));

kernel void foo1(float2 in, global float2 *out) { *out = in + 0.5;} // expected-error {{scalar operand type has greater rank than the type of the vector element. ('float2' (vector of 2 'float' values) and 'double')}}

kernel void foo2(float2 in, global float2 *out) { *out = 0.5 + in;} // expected-error {{scalar operand type has greater rank than the type of the vector element. ('double' and 'float2' (vector of 2 'float' values))}}

kernel void foo3(float2 in, global float2 *out) { *out = 0.5f + in;}

kernel void foo4(long2 in, global long2 *out) { *out = 5 + in;}

kernel void foo5(float2 in, global float2 *out) {
    float* f;
    *out = f + in; // expected-error{{cannot convert between vector and non-scalar values ('__private float *' and 'float2' (vector of 2 'float' values))}}
}

kernel void foo6(int2 in, global int2 *out) {
    int* f;
    *out = f + in; // expected-error{{cannot convert between vector and non-scalar values ('__private int *' and 'int2' (vector of 2 'int' values))}}
}
