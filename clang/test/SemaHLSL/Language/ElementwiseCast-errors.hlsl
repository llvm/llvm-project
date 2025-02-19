// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -verify

export void cantCast() {
  int A[3] = {1,2,3};
  int B[4] = {1,2,3,4};
  B = (int[4])A;
  // expected-error@-1 {{C-style cast from 'int *' to 'int[4]' is not allowed}}
}

struct S {
// expected-note@-1 {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int2' (aka 'vector<int, 2>') to 'const S' for 1st argument}}
// expected-note@-2 {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'int2' (aka 'vector<int, 2>') to 'S' for 1st argument}}
// expected-note@-3 {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}
  int A : 8;
  int B;
};

// casting types which contain bitfields is not yet supported.
export void cantCast2() {
  S s = {1,2};
  int2 C = (int2)s;
  // expected-error@-1 {{cannot convert 'S' to 'int2' (aka 'vector<int, 2>') without a conversion operator}}
}

export void cantCast3() {
  int2 C = {1,2};
  S s = (S)C;
  // expected-error@-1 {{no matching conversion for C-style cast from 'int2' (aka 'vector<int, 2>') to 'S'}}
}
