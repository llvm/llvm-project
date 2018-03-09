// RUN: %clang_cc1 -fsyntax-only -Wunused -fdouble-square-bracket-attributes -verify %s

struct [[maybe_unused]] S1 { // ok
  int a [[maybe_unused]];
};
struct [[maybe_unused maybe_unused]] S2 { // expected-error {{attribute 'maybe_unused' cannot appear multiple times in an attribute specifier}}
  int a;
};
struct [[maybe_unused("Wrong")]] S3 { // expected-error {{'maybe_unused' cannot have an argument list}}
  int a;
};

