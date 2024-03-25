// RUN: %clang_cc1 -x c++ -fsyntax-only -verify %s -Wenum-compare
// expected-no-diagnostics

enum E1 {
  E11 = 0
};

enum E2 {
  E21 = 0,
  E22 = E11,
  E23 = E21 + E22
};
