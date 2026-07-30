// RUN: %clang_cc1 -fsyntax-only -Wunused -std=c2x -verify %s

struct [[maybe_unused]] S1 { // ok
  int a [[maybe_unused]];
};
struct [[maybe_unused, maybe_unused]] S2 { // ok
  int a;
};
struct [[maybe_unused("Wrong")]] S3 { // expected-error {{'maybe_unused' cannot have an argument list}}
  int a;
};

void func(void) {
  int a[10] [[maybe_unused]]; // expected-error {{'maybe_unused' attribute cannot be applied to types}}
}
