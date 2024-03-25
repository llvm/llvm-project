// RUN: %clang_cc1 -std=c++2c -fsyntax-only -verify %s

struct S {
  void f() = delete("deleted (1)"); // expected-note {{explicitly marked deleted}}

  template <typename T>
  T g() = delete("deleted (2)"); // expected-note {{explicitly deleted}}
};

template <typename T>
struct TS {
  T f() = delete("deleted (3)"); // expected-note {{explicitly marked deleted}}

  template <typename U>
  T g(U) = delete("deleted (4)"); // expected-note {{explicitly deleted}}
};

void f() = delete("deleted (5)"); // expected-note {{explicitly deleted}}

template <typename T>
T g() = delete("deleted (6)"); // expected-note {{explicitly deleted}}

void h() {
  S{}.f(); // expected-error {{attempt to use a deleted function: deleted (1)}}
  S{}.g<int>(); // expected-error {{call to deleted member function 'g': deleted (2)}}
  TS<int>{}.f(); // expected-error {{attempt to use a deleted function: deleted (3)}}
  TS<int>{}.g<int>(0); // expected-error {{call to deleted member function 'g': deleted (4)}}
  f(); // expected-error {{call to deleted function 'f': deleted (5)}}
  g<int>(); // expected-error {{call to deleted function 'g': deleted (6)}}
}
