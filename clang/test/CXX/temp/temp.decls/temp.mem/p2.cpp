// RUN: %clang_cc1 -std=c++14 -fsyntax-only -verify %s

template <typename>
void quux();

void fun() {
  struct foo {
    template <typename> struct bar {};  // expected-error{{templates cannot be declared inside of a local class}}
    template <typename> void baz() {}   // expected-error{{templates cannot be declared inside of a local class}}
    template <typename> void qux();     // expected-error{{templates cannot be declared inside of a local class}}
    template <typename> using corge = int; // expected-error{{templates cannot be declared inside of a local class}}
    template <typename T> static T grault; // expected-error{{templates cannot be declared inside of a local class}}
  };
}
