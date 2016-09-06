// RUN: %clang_cc1 -fsyntax-only -Wunused -Wused-but-marked-unused -std=c++1z -Wc++1z-extensions -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wunused -Wused-but-marked-unused -std=c++11 -Wc++1z-extensions -verify -DEXT %s

static_assert(__has_cpp_attribute(maybe_unused) == 201603, "");

struct [[maybe_unused]] S {};

void f() {
  int x; // expected-warning {{unused variable}}
  typedef int I; // expected-warning {{unused typedef 'I'}}

  // Should not warn about these due to not being used.
  [[maybe_unused]] int y;
  typedef int maybe_unused_int [[maybe_unused]];

  // Should not warn about these uses.
  S s;
  maybe_unused_int test;
  y = 12;
}

#ifdef EXT
// expected-warning@6 {{use of the 'maybe_unused' attribute is a C++1z extension}}
// expected-warning@13 {{use of the 'maybe_unused' attribute is a C++1z extension}}
// expected-warning@14 {{use of the 'maybe_unused' attribute is a C++1z extension}}
#endif
