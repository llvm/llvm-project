// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -verify

struct S {
// expected-note@-1 {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int' to 'const S' for 1st argument}}
// expected-note@-2 {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'int' to 'S' for 1st argument}}
// expected-note@-3 {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}
  int A : 8;
  int B;
};

struct R {
// expected-note@-1 {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int' to 'const R' for 1st argument}}
// expected-note@-2 {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'int' to 'R' for 1st argument}}
// expected-note@-3 {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}
  int A;
  union {
    float F;
    int4 G;
  };
};

// casting types which contain bitfields is not yet supported.
export void cantCast() {
  S s = (S)1;
  // expected-error@-1 {{no matching conversion for C-style cast from 'int' to 'S'}}
}

// Can't cast a union
export void cantCast2() {
  R r = (R)1;
  // expected-error@-1 {{no matching conversion for C-style cast from 'int' to 'R'}}
}
