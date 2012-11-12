// RUN: %clang_cc1 -fsyntax-only -verify %s -fno-lax-vector-conversions

typedef float float4 __attribute__((vector_size(16)));
typedef int int4 __attribute__((vector_size(16)));
typedef int4* int4p;

void test1(float4 a, int4 *result, int i) {
    result[i] = a; // expected-error {{assigning to 'int4' from incompatible type 'float4'}}
}

void test2(float4 a, int4p result, int i) {
    result[i] = a; // expected-error {{assigning to 'int4' from incompatible type 'float4'}}
}

// PR2039
typedef int a[5];
void test3() {
  typedef const a b;
  b r;
  r[0]=10;  // expected-error {{read-only variable is not assignable}}
}

int test4(const a y) {
  y[0] = 10; // expected-error {{read-only variable is not assignable}}
}

