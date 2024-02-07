// RUN: %clang_cc1 -std=c++23 -x c++ %s -verify
// RUN: %clang_cc1 -std=c++20 -pedantic -x c++ %s -verify=ext,expected

struct A{};
struct B{ explicit operator bool() { return true; } };

template <bool cond>
void f() {
  [[assume(cond)]]; // ext-warning {{C++23 extension}}
}

template <bool cond>
struct S {
  void f() {
    [[assume(cond)]]; // ext-warning {{C++23 extension}}
  }
};

bool f2();

void g(int x) {
  f<true>();
  f<false>();
  S<true>{}.f();
  S<false>{}.f();
  [[assume(f2())]]; // expected-warning {{side effects that will be discarded}} ext-warning {{C++23 extension}}

  [[assume((x = 3))]]; // expected-warning {{has side effects that will be discarded}} // ext-warning {{C++23 extension}}
  [[assume(x++)]]; // expected-warning {{has side effects that will be discarded}} // ext-warning {{C++23 extension}}
  [[assume(++x)]]; // expected-warning {{has side effects that will be discarded}} // ext-warning {{C++23 extension}}
  [[assume([]{ return true; }())]]; // expected-warning {{has side effects that will be discarded}} // ext-warning {{C++23 extension}}
  [[assume(B{})]]; // expected-warning {{has side effects that will be discarded}} // ext-warning {{C++23 extension}}
  [[assume((1, 2))]]; // expected-warning {{has no effect}} // ext-warning {{C++23 extension}}

  [[assume]]; // expected-error {{takes one argument}}
  [[assume(z)]]; // expected-error {{undeclared identifier}}
  [[assume(A{})]]; // expected-error {{not contextually convertible to 'bool'}}
  [[assume(true)]] if (true) {} // expected-error {{only applies to empty statements}}
  [[assume(true)]] {} // expected-error {{only applies to empty statements}}
  [[assume(true)]] for (;false;) {} // expected-error {{only applies to empty statements}}
  [[assume(true)]] while (false) {} // expected-error {{only applies to empty statements}}
  [[assume(true)]] label:; // expected-error {{cannot be applied to a declaration}}
  [[assume(true)]] goto label; // expected-error {{only applies to empty statements}}
}

// Check that 'x' is ODR-used here.
constexpr int h(int x) { return sizeof([=] { [[assume(x)]]; }); } // ext-warning {{C++23 extension}}
static_assert(h(4) == sizeof(int));

static_assert(__has_cpp_attribute(assume));
