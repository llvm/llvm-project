// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// C++03 requires that we check for a copy constructor when binding a
// reference to a reference-compatible rvalue, since we are allowed to
// make a copy. C++0x does not permit the copy, so ensure that we
// don't diagnose cases where the copy constructor is unavailable.

struct X1 {
  X1();
  explicit X1(const X1&);
};

struct X2 {
  X2();

private:
  X2(const X2&);
};

struct X3 {
  X3();

private:
  X3(X3&);
};

template<typename T>
T get_value_badly() {
  double *dp = 0;
  T *tp = dp;
  return T();
}

template<typename T>
struct X4 {
  X4();
  X4(const X4&, T = get_value_badly<T>());
};

void g1(const X1&);
void g2(const X2&);
void g3(const X3&);
void g4(const X4<int>&);

void test() {
  g1(X1());
  g2(X2());
  g3(X3());
  g4(X4<int>());
}

// Check that unavailable copy constructors do not cause SFINAE failures.
template<int> struct int_c { };

template<typename T> T f(const T&);

template<typename T>
int &g(int_c<sizeof(f(T()))> * = 0);  // expected-note{{candidate function [with T = X3]}}

template<typename T> float &g();  // expected-note{{candidate function [with T = X3]}}

void h() {
  float &fp = g<X3>();  // expected-error{{call to 'g' is ambiguous}}
}
