// RUN: %clang_cc1 -std=c++20 -verify=both,expected %s -Wexit-time-destructors -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -std=c++20 -verify=both,ref      %s -Wexit-time-destructors

// both-no-diagnostics

struct S {
  int a;
  constexpr S() {}
  constexpr ~S() {
  }
};
S s{};


