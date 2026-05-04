// Test that we can successfully compile this code, especially under ASAN.
// RUN: %clang_cc1 -verify -std=c++20 -fsyntax-only %s
// expected-no-diagnostics
struct Foo {
  Foo* f;
  operator bool() const { return true; }
};
constexpr Foo f((Foo[]){});
int foo() {
  if (Foo(*f.f)) return 1;
  return 0;
}
