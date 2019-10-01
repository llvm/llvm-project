// RUN: %clang_cc1 -fsyntax-only -Wunused -std=c2x -verify %s

struct [[maybe_unused]] S1 { // ok
  int a [[maybe_unused]];
};

enum [[maybe_unused]] E1 {
  EnumVal [[maybe_unused]]
};

[[maybe_unused]] void unused_func([[maybe_unused]] int parm) {
  typedef int maybe_unused_int [[maybe_unused]];
  [[maybe_unused]] int I;
}

void f1(void) {
  int x; // expected-warning {{unused variable}}
  typedef int I; // expected-warning {{unused typedef 'I'}}

  // Should not warn about these due to not being used.
  [[maybe_unused]] int y;
  typedef int maybe_unused_int [[maybe_unused]];

  // Should not warn about these uses.
  struct S1 s;
  maybe_unused_int test;
  y = 12;
}

void f2(void);
[[maybe_unused]] void f2(void);

void f2(void) {
}

