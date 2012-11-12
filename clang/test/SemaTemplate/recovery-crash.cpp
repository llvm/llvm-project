// RUN: %clang_cc1 -fsyntax-only -verify %s

// Clang used to crash trying to recover while adding 'this->' before Work(x);

template <typename> struct A {
  static void Work(int);  // expected-note{{must qualify identifier}}
};

template <typename T> struct B : public A<T> {
  template <typename T2> B(T2 x) {
    Work(x);  // expected-error{{use of undeclared identifier}}
  }
};

void Test() {
  B<int> b(0);  // expected-note{{in instantiation of function template}}
}

