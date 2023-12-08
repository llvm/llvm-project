// RUN: %clang_cc1 %s -pedantic -verify -triple=x86_64-apple-darwin9

typedef float float4 __attribute__((ext_vector_type(4)));

float test_fminf(float f, int i, int *ptr, float4 v) {
  float r1 = __builtin_fminf(f, ptr);
  // expected-error@-1 {{passing 'int *' to parameter of incompatible type 'float'}}
  float r2 = __builtin_fminf(ptr, f);
  // expected-error@-1 {{passing 'int *' to parameter of incompatible type 'float'}}
  float r3 = __builtin_fminf(v, f);
  // expected-error@-1 {{passing 'float4' (vector of 4 'float' values) to parameter of incompatible type 'float'}}
  float r4 = __builtin_fminf(f, v);
  // expected-error@-1 {{passing 'float4' (vector of 4 'float' values) to parameter of incompatible type 'float'}}


  int *r5 = __builtin_fminf(f, f);
  // expected-error@-1 {{initializing 'int *' with an expression of incompatible type 'float'}}

  int *r6 = __builtin_fminf(f, v);
  // expected-error@-1 {{passing 'float4' (vector of 4 'float' values) to parameter of incompatible type 'float'}}
}
