// RUN: %clang_cc1 -fsyntax-only -Wunused-lambda-capture -Wused-but-marked-unused -Wno-uninitialized -verify -std=c++20 %s

void test() {
  int i;
  auto explicit_by_value_unused_requires = [i] (auto) requires requires { i; } {}; // expected-warning{{lambda capture 'i' is not required to be captured for this use}}
  explicit_by_value_unused_requires(1);
}
