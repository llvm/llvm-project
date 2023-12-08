// RUN: %clang_cc1 -fsyntax-only -verify %s

void f1();

struct X {
  void f2();
};

struct Y {
  friend void ::f1() { } // expected-error{{friend function definition cannot be qualified with '::'}}
  friend void X::f2() { } // expected-error{{friend function definition cannot be qualified with 'X::'}}
};

template <typename T> struct Z {
  friend void T::f() {} // expected-error{{friend function definition cannot be qualified with 'T::'}}
};

void local() {
  void f();

  struct Local {
    friend void f() { } // expected-error{{friend function cannot be defined in a local class}}
  };
}
