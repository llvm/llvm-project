// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

consteval int Fun() { return; } // expected-error {{non-void consteval function 'Fun' should return a value}}

template <typename T> consteval int FunT1() { return; } // expected-error {{non-void consteval function 'FunT1' should return a value}}
template <typename T> consteval int FunT2() { return 0; }
template <> consteval int FunT2<double>() { return 0; }
template <> consteval int FunT2<int>() { return; } // expected-error {{non-void consteval function 'FunT2<int>' should return a value}}

enum E {};

constexpr E operator+(E,E) { return; }	// expected-error {{non-void constexpr function 'operator+' should return a value}}
consteval E operator+(E,E) { return; }  // expected-error {{non-void consteval function 'operator+' should return a value}}
template <typename T> constexpr E operator-(E,E) { return; } // expected-error {{non-void constexpr function 'operator-' should return a value}}
template <typename T> consteval E operator-(E,E) { return; } // expected-error {{non-void consteval function 'operator-' should return a value}}

template <typename T> constexpr E operator*(E,E);
template <typename T> consteval E operator/(E,E);
template <> constexpr E operator*<int>(E,E) { return; } // expected-error {{non-void constexpr function 'operator*<int>' should return a value}}
template <> consteval E operator/<int>(E,E) { return; } // expected-error {{non-void consteval function 'operator/<int>' should return a value}}

consteval void no_return() {}
consteval void with_return() { return; }
consteval void with_return_void() { return void(); }
void use_void_fn() {
  no_return();
  with_return();
  with_return_void();
}
