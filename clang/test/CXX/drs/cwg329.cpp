// RUN: %clang_cc1 -std=c++98 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify
// RUN: %clang_cc1 -std=c++11 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify
// RUN: %clang_cc1 -std=c++14 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify
// RUN: %clang_cc1 -std=c++17 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify
// RUN: %clang_cc1 -std=c++20 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify
// RUN: %clang_cc1 -std=c++23 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify
// RUN: %clang_cc1 -std=c++2c %s -fexceptions -fcxx-exceptions -pedantic-errors -verify

namespace cwg329 { // cwg329: 3.5
  struct B {};
  template<typename T> struct A : B {
    friend void f(A a) { g(a); }
    friend void h(A a) { g(a); }
    // expected-error@-1 {{use of undeclared identifier 'g'}}
    //   expected-note@#cwg329-h-call {{in instantiation of member function 'cwg329::h' requested here}}
    friend void i(B b) {} // #cwg329-i
    // expected-error@-1 {{redefinition of 'i'}}
    //   expected-note@#cwg329-b {{in instantiation of template class 'cwg329::A<char>' requested here}}
    //   expected-note@#cwg329-i {{previous definition is here}}
  };
  A<int> a;
  A<char> b; // #cwg329-b

  void test() {
    h(a); // #cwg329-h-call
  }
} // namespace cwg329
