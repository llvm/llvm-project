// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// The result of the expression const_cast<T>(v) is of type T. If T is
// an lvalue reference to object type, the result is an lvalue; if T
// is an rvalue reference to object type, the result is an xvalue;.

unsigned int f(int);

struct X {};

template<typename T> T& lvalue();
template<typename T> T&& xvalue();
template<typename T> T prvalue();

void test_classification(const int *ptr, X x) {
  int *&&ptr0 = const_cast<int *&&>(ptr);
  int *&&ptr1 = const_cast<int *&&>(xvalue<const int*>());
  int *&&ptr2 = const_cast<int *&&>(prvalue<const int*>()); // expected-error {{const_cast from rvalue to reference type 'int *&&'}}
  X &&ptr3 = const_cast<X&&>(x);
  X &&ptr4 = const_cast<X&&>(xvalue<X>());
  X &&ptr5 = const_cast<X&&>(prvalue<X>());

  int *&ptr6 = const_cast<int *&>(ptr);
  int *&ptr7 = const_cast<int *&>(xvalue<const int*>()); // expected-error {{const_cast from rvalue to reference type 'int *&'}}
  int *&ptr8 = const_cast<int *&>(prvalue<const int*>()); // expected-error {{const_cast from rvalue to reference type 'int *&'}}
  X &ptr9 = const_cast<X&>(x);
  X &ptrA = const_cast<X&>(xvalue<X>()); // expected-error {{const_cast from rvalue to reference type 'X &'}}
  X &ptrB = const_cast<X&>(prvalue<X>()); // expected-error {{const_cast from rvalue to reference type 'X &'}}
}

struct A {
  volatile unsigned ubf : 4;
  volatile unsigned uv;
  volatile int sv;
  void foo();
  bool pred();
};

void test(A &a) {
  unsigned &t0 = const_cast<unsigned&>(a.ubf); // expected-error {{const_cast from bit-field lvalue to reference type}}
  unsigned &t1 = const_cast<unsigned&>(a.foo(), a.ubf); // expected-error {{const_cast from bit-field lvalue to reference type}}
  unsigned &t2 = const_cast<unsigned&>(a.pred() ? a.ubf : a.ubf); // expected-error {{const_cast from bit-field lvalue to reference type}}
  unsigned &t3 = const_cast<unsigned&>(a.pred() ? a.ubf : a.uv); // expected-error {{const_cast from bit-field lvalue to reference type}}
  unsigned &t4 = const_cast<unsigned&>(a.pred() ? a.ubf : a.sv); // expected-error {{const_cast from rvalue to reference type}}
}
