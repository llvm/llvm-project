// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s


// If P is an rvalue reference to a cv-unqualified template parameter
// and the argument is an lvalue, the type "lvalue reference to A" is
// used in place of A for type deduction.
template<typename T> struct X { };

template<typename T> X<T> f0(T&&);

struct Y { };

template<typename T> T prvalue();
template<typename T> T&& xvalue();
template<typename T> T& lvalue();

void test_f0() {
  X<int> xi0 = f0(prvalue<int>());
  X<int> xi1 = f0(xvalue<int>());
  X<int&> xi2 = f0(lvalue<int>());
  X<Y> xy0 = f0(prvalue<Y>());
  X<Y> xy1 = f0(xvalue<Y>());
  X<Y&> xy2 = f0(lvalue<Y>());
}

template<typename T> X<T> f1(const T&&); // expected-note{{candidate function [with T = int] not viable: no known conversion from 'int' to 'const int &&' for 1st argument}} \
// expected-note{{candidate function [with T = Y] not viable: no known conversion from 'Y' to 'const Y &&' for 1st argument}}

void test_f1() {
  X<int> xi0 = f1(prvalue<int>());
  X<int> xi1 = f1(xvalue<int>());
  f1(lvalue<int>()); // expected-error{{no matching function for call to 'f1'}}
  X<Y> xy0 = f1(prvalue<Y>());
  X<Y> xy1 = f1(xvalue<Y>());
  f1(lvalue<Y>()); // expected-error{{no matching function for call to 'f1'}}
}

namespace std_example {
  template <class T> int f(T&&); 
  template <class T> int g(const T&&); // expected-note{{candidate function [with T = int] not viable: no known conversion from 'int' to 'const int &&' for 1st argument}}

  int i;
  int n1 = f(i);
  int n2 = f(0);
  int n3 = g(i); // expected-error{{no matching function for call to 'g'}}
}
