// RUN: %clang_cc1 -fsyntax-only -verify %s

template <class T> struct A;
template <class T> struct B {
  void f();
};

void t1() {
  struct S {
    template <class T> friend void f();
    // expected-error@-1 {{templates can only be declared in namespace or class scope}}
  };
}

void t2() {
  struct S {
    template <class T> friend struct A;
    // expected-error@-1 {{templates cannot be declared inside of a local class}}
  };
}

void t3() {
  struct S {
    template <class T> friend void B<T>::f();
    // expected-error@-1 {{templates cannot be declared inside of a local class}}
  };
}
