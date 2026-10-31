// RUN: %clang_cc1 -fsyntax-only -Wall -verify %s

template<typename T> struct A {
  A() : a(1) { } // expected-error{{cannot initialize a member subobject of type 'void *' with an rvalue of type 'int'}}

  T a;
};

A<int> a0;
A<void*> a1; // expected-note{{in instantiation of member function 'A<void *>::A' requested here}}

template<typename T> struct B {
  B() : b(1), // expected-warning {{field 'b' will be initialized after field 'a'}}
    a(2) { }
  
  int a;
  int b;
};

B<int> b0; // expected-note {{in instantiation of member function 'B<int>::B' requested here}}

template <class T> struct AA { AA(int); };
template <class T> class BB : public AA<T> {
public:
  BB() : AA<T>(1) {}
};
BB<int> x;

struct X {
  X();
};
template<typename T>
struct Y {
  Y() : x() {}
  X x;
};
Y<int> y;

template<typename T> struct Array {
  int a[3];
  Array() : a() {}
};
Array<int> s;

namespace GH194986 {
  struct S {};
  template <typename T> struct SS { T t1; T t2; }; // expected-note {{candidate template ignored: could not match 'GH194986::SS<T>' against 'const char *'}} expected-note {{implicit deduction guide declared as 'template <typename T> SS(GH194986::SS<T>) -> GH194986::SS<T>'}} expected-note {{candidate function template not viable: requires 0 arguments, but 1 was provided}} expected-note {{implicit deduction guide declared as 'template <typename T> SS() -> GH194986::SS<T>'}}
  template <class T, class... Args> T C(Args... args) { return SS("foo"); } // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'SS'}}
  S s = C<S>();

  template <class T> struct SS2 { T t1, t2; }; // expected-note {{candidate template ignored: could not match 'GH194986::SS2<T>' against 'const char *'}} expected-note {{implicit deduction guide declared as 'template <class T> SS2(GH194986::SS2<T>) -> GH194986::SS2<T>'}} expected-note {{candidate function template not viable: requires 0 arguments, but 1 was provided}} expected-note {{implicit deduction guide declared as 'template <class T> SS2() -> GH194986::SS2<T>'}}
  template <class> void C2() {
    SS2("foo"); // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'SS2'}}
  }
  template void C2<int>();
};
