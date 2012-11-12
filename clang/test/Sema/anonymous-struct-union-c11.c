// Check for warnings in non-C11 mode:
// RUN: %clang_cc1 -fsyntax-only -verify -Wc11-extensions %s

// Expect no warnings in C11 mode:
// RUN: %clang_cc1 -fsyntax-only -pedantic -Werror -std=c11 %s

struct s {
  int a;
  struct { // expected-warning{{anonymous structs are a C11 extension}}
    int b;
  };
};

struct t {
  int a;
  union { // expected-warning{{anonymous unions are a C11 extension}}
    int b;
  };
};
