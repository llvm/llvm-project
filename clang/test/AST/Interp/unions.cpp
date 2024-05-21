// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -verify=ref,both %s

// both-no-diagnostics

union U {
  int a;
  int b;
};

constexpr U a = {12};
static_assert(a.a == 12, "");


