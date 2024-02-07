// RUN: %clang_cc1 -std=c++23 -x c++ %s -verify
// RUN: %clang_cc1 -std=c++20 -pedantic -x c++ %s -verify=ext
// expected-no-diagnostics

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

void g() {
  f<true>();
  f<false>();
  S<true>{}.f();
  S<false>{}.f();
}

// Check that 'x' is ODR-used here.
constexpr int h(int x) { return sizeof([=] { [[assume(x)]]; }); } // ext-warning {{C++23 extension}}
static_assert(h(4) == sizeof(int));
