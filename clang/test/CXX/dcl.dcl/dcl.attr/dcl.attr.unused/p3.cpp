// RUN: %clang_cc1 -fsyntax-only -Wunused -Wused-but-marked-unused -std=c++17 -Wc++17-extensions -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wunused -Wused-but-marked-unused -std=c++11 -Wc++17-extensions -verify -DEXT %s

static_assert(__has_cpp_attribute(maybe_unused) == 201603, "");

struct [[maybe_unused]] S {};

enum E1 {
  EnumVal [[maybe_unused]],
  UsedEnumVal,
};

void f() {
  int x; // expected-warning {{unused variable}}
  typedef int I; // expected-warning {{unused typedef 'I'}}
  E1 e;
  switch (e) { // expected-warning {{enumeration value 'UsedEnumVal' not handled in switch}}
  }

  // Should not warn about these due to not being used.
  [[maybe_unused]] int y;
  typedef int maybe_unused_int [[maybe_unused]];

  // Should not warn about these uses.
  S s;
  maybe_unused_int test;
  y = 12;
  switch (e) {
  case UsedEnumVal:
    break;
  }
}

#ifdef EXT
// expected-warning@6 {{use of the 'maybe_unused' attribute is a C++17 extension}}
// expected-warning@9 {{use of the 'maybe_unused' attribute is a C++17 extension}}
// expected-warning@9 {{attributes on an enumerator declaration are a C++17 extension}}
// expected-warning@21 {{use of the 'maybe_unused' attribute is a C++17 extension}}
// expected-warning@22 {{use of the 'maybe_unused' attribute is a C++17 extension}}
#endif
