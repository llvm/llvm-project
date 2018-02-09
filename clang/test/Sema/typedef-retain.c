// RUN: %clang_cc1 -fsyntax-only -verify %s -fno-lax-vector-conversions

typedef float float4 __attribute__((vector_size(16)));
typedef int int4 __attribute__((vector_size(16)));
typedef int4* int4p;

void test1(float4 a, int4 *result, int i) {
    result[i] = a; // expected-error {{assigning to 'int4' (vector of 4 'int' values) from incompatible type 'float4' (vector of 4 'float' values)}}
}

void test2(float4 a, int4p result, int i) {
    result[i] = a; // expected-error {{assigning to 'int4' (vector of 4 'int' values) from incompatible type 'float4' (vector of 4 'float' values)}}
}

// PR2039
typedef int a[5];
void test3() {
  typedef const a b;
  b r;       // expected-note {{variable 'r' declared const here}}
  r[0] = 10; // expected-error {{cannot assign to variable 'r' with const-qualified type 'b' (aka 'int const[5]')}}
}

int test4(const a y) {
  y[0] = 10; // expected-error {{read-only variable is not assignable}}
}

