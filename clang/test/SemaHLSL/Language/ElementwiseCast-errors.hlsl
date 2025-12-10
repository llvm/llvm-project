// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -verify

export void cantCast() {
  int A[3] = {1,2,3};
  int B[4] = {1,2,3,4};
  B = (int[4])A;
  // expected-error@-1 {{C-style cast from 'int[3]' to 'int[4]' is not allowed}}
}

struct R {
// expected-note@-1 {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int2' (aka 'vector<int, 2>') to 'const R' for 1st argument}}
// expected-note@-2 {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'int2' (aka 'vector<int, 2>') to 'R' for 1st argument}}
// expected-note@-3 {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}
  int A;
  union {
    float F;
    int4 G;
  };
};

export void cantCast4() {
  int2 A = {1,2};
  R r = R(A);
  // expected-error@-1 {{no matching conversion for functional-style cast from 'int2' (aka 'vector<int, 2>') to 'R'}}
  R r2;
  r2.A = 1;
  r2.F = 2.0;
  int2 B = (int2)r2;
  // expected-error@-1 {{cannot convert 'R' to 'int2' (aka 'vector<int, 2>') without a conversion operator}}
}
