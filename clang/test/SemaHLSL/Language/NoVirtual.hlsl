// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -verify %s

struct Base {
  int X;
  void MemberFunction();  // valid
  virtual void MemberFunction2(); // expected-error{{virtual functions are unsupported in HLSL}}
};

struct Derived : virtual Base { // expected-error{{virtual inheritance is unsupported in HLSL}}
  int Y;

  void MemberFunction2() override; // expected-error{{only virtual member functions can be marked 'override'}}
};

