// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -verify

struct Base {
  int A, B;
};

struct Derived: Base {
  float F, G;
};

struct Other {
  int C, D;
};

export void fn() {
  Base C = {5,6};

  Other O = {7,8};
  // expected-error@+1{{assigning to 'Base' from incompatible type 'Other'}}
  C = O;

  int2 I2 = {9,10};
  // expected-error@+1{{assigning to 'Base' from incompatible type 'int2' (aka 'vector<int, 2>')}}
  C = I2;

  Derived D = {1,2,3,4};
  // expected-error@+1{{assigning to 'Base' from incompatible type 'Derived'}}
  C = D;
}
