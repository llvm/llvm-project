// RUN: %clang_cc1 -std=c11 -verify %s

struct GH31295 {
  struct { int x; };
  int arr[sizeof(x)]; // expected-error{{use of undeclared identifier 'x'}}
};
