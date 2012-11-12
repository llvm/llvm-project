// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

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

  int &operator+(const X0&) &;
  float &operator+(const X0&) &&;

  template<typename T> int &operator+(const T&) &;
  template<typename T> float &operator+(const T&) &&;

  int &h() const&;
  float &h() &&;
  int &h2() const&;
  float &h2() const&&;
};

void X0::g() {
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
