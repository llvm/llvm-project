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

template<typename T> void f3(T);

namespace N {
  template<typename T> void f4(T);
}

template<typename T> struct A {
  friend void f3(T) {}
  friend void f3<T>(T) {} // expected-error{{friend function specialization cannot be defined}}
  friend void N::f4(T) {} // expected-error{{friend function definition cannot be qualified with 'N::'}}
  friend void N::f4<T>(T) {} // expected-error{{friend function definition cannot be qualified with 'N::'}}
};
