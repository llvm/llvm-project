// RUN: %clang_cc1 -fsycl-is-device -std=c++20 -verify -fsyntax-only %s

// In C++20 a reference to an array of unknown bound binds to an array of known
// bound, so a C-style cast to it cannot convert between address spaces that
// do not overlap either.
void f(int [[clang::sycl_global]] (&g)[4]) {
  (void)(int [[clang::sycl_global]] (&)[])g;
  (void)(int [[clang::sycl_local]] (&)[])g;  // expected-error {{C-style cast from '[[clang::sycl_global]] int[4]' to '[[clang::sycl_local]] int (&)[]' converts between mismatching address spaces}}
  (void)(int [[clang::sycl_local]] (&)[4])g; // expected-error {{C-style cast from '[[clang::sycl_global]] int[4]' to '[[clang::sycl_local]] int (&)[4]' converts between mismatching address spaces}}
}
