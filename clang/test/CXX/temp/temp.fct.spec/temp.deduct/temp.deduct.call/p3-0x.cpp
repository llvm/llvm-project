// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++1z -fsyntax-only -verify %s

// A forwarding reference is an rvalue reference to a cv-unqualified template
// parameter that does not represent a template parameter of a class template.
#if __cplusplus > 201402L
namespace ClassTemplateParamNotForwardingRef {
  // This is not a forwarding reference.
  template<typename T> struct A { // expected-note {{candidate}}
    A(T&&); // expected-note {{expects an rvalue}}
  };
  int n;
  A a = n; // expected-error {{no viable constructor or deduction guide}}

  A b = 0;
  A<int> *pb = &b;

  // This is a forwarding reference.
  template<typename T> A(T&&) -> A<T>;
  A c = n;
  A<int&> *pc = &c;

  A d = 0;
  A<int> *pd = &d;

  template<typename T = void> struct B {
    // This is a forwarding reference.
    template<typename U> B(U &&);
  };
  B e = n;
  B<void> *pe = &e;
}
#endif

// If P is a forwarding reference and the argument is an lvalue, the type
// "lvalue reference to A" is used in place of A for type deduction.
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

template<typename T> X<T> f1(const T&&); // expected-note{{candidate function [with T = int] not viable: expects an rvalue for 1st argument}} \
// expected-note{{candidate function [with T = Y] not viable: expects an rvalue for 1st argument}}

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
  template <class T> int g(const T&&); // expected-note{{candidate function [with T = int] not viable: expects an rvalue for 1st argument}}

  int i;
  int n1 = f(i);
  int n2 = f(0);
  int n3 = g(i); // expected-error{{no matching function for call to 'g'}}

#if __cplusplus > 201402L
  template<class T> struct A { // expected-note {{candidate}}
    template<class U>
    A(T &&, U &&, int *); // expected-note {{[with T = int, U = int] not viable: expects an rvalue}}
    A(T &&, int *);       // expected-note {{requires 2}}
  };
  template<class T> A(T &&, int *) -> A<T>; // expected-note {{requires 2}}

  int *ip;
  A a{i, 0, ip};  // expected-error {{no viable constructor or deduction guide}}
  A a0{0, 0, ip};
  A a2{i, ip};

  A<int> &a0r = a0;
  A<int&> &a2r = a2;
#endif
}
