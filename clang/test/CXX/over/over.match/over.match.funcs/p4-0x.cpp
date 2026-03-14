// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

template<typename T> T &lvalue();
template<typename T> T &&xvalue();
template<typename T> T prvalue();

struct X0 {
  int &f() &;
  float &f() &&;

  template<typename T> int &ft(T) &;
  template<typename T> float &ft(T) &&;

  typedef int &(*func_int_ref)();
  typedef float &(*func_float_ref)();

  operator func_int_ref() &;
  operator func_float_ref() &&;

  void g();

  void c() const; // expected-note {{'c' declared here}}
  void v() volatile; // expected-note {{'v' declared here}}
  void r() __restrict__; // expected-note {{'r' declared here}}
  void cr() const __restrict__; // expected-note {{'cr' declared here}}
  void cv() const volatile;
  void vr() volatile __restrict__; // expected-note {{'vr' declared here}}
  void cvr() const volatile __restrict__;

  void lvalue() &; // expected-note 2 {{'lvalue' declared here}}
  void const_lvalue() const&;
  void rvalue() &&; // expected-note {{'rvalue' declared here}}

  int &operator+(const X0&) &;
  float &operator+(const X0&) &&;

  template<typename T> int &operator+(const T&) &;
  template<typename T> float &operator+(const T&) &&;

  int &h() const&;
  float &h() &&;
  int &h2() const&;
  float &h2() const&&;
};

void X0::g() { // expected-note {{'g' declared here}}
  int &ir1 = f();
  int &ir2 = X0::f();
}

void test_ref_qualifier_binding() {
  int &ir1 = lvalue<X0>().f();
  float &fr1 = xvalue<X0>().f();
  float &fr2 = prvalue<X0>().f();
  int &ir2 = lvalue<X0>().ft(1);
  float &fr3 = xvalue<X0>().ft(2);
  float &fr4 = prvalue<X0>().ft(3);
}

void test_ref_qualifier_binding_with_surrogates() {
  int &ir1 = lvalue<X0>()();
  float &fr1 = xvalue<X0>()();
  float &fr2 = prvalue<X0>()();
}

void test_ref_qualifier_binding_operators() {
  int &ir1 = lvalue<X0>() + prvalue<X0>();
  float &fr1 = xvalue<X0>() + prvalue<X0>();
  float &fr2 = prvalue<X0>() + prvalue<X0>();
  int &ir2 = lvalue<X0>() + 1;
  float &fr3 = xvalue<X0>() + 2;
  float &fr4 = prvalue<X0>() + 3;
}

void test_ref_qualifier_overloading() {
  int &ir1 = lvalue<X0>().h();
  float &fr1 = xvalue<X0>().h();
  float &fr2 = prvalue<X0>().h();
  int &ir2 = lvalue<X0>().h2();
  float &fr3 = xvalue<X0>().h2();
  float &fr4 = prvalue<X0>().h2();
}

void test_diagnostics(const volatile X0 &__restrict__ cvr) {
  cvr.g(); // expected-error {{'this' argument to member function 'g' has type 'const volatile X0', but function is not marked const or volatile}}
  cvr.c(); // expected-error {{not marked volatile}}
  cvr.v(); // expected-error {{not marked const}}
  cvr.r(); // expected-error {{not marked const or volatile}}
  cvr.cr(); // expected-error {{not marked volatile}}
  cvr.cv();
  cvr.vr(); // expected-error {{not marked const}}
  cvr.cvr();

  lvalue<X0>().lvalue();
  lvalue<X0>().const_lvalue();
  lvalue<X0>().rvalue(); // expected-error {{'this' argument to member function 'rvalue' is an lvalue, but function has rvalue ref-qualifier}}

  xvalue<X0>().lvalue(); // expected-error {{'this' argument to member function 'lvalue' is an rvalue, but function has non-const lvalue ref-qualifier}}
  xvalue<X0>().const_lvalue();
  xvalue<X0>().rvalue();

  prvalue<X0>().lvalue(); // expected-error {{'this' argument to member function 'lvalue' is an rvalue, but function has non-const lvalue ref-qualifier}}
  prvalue<X0>().const_lvalue();
  prvalue<X0>().rvalue();
}
