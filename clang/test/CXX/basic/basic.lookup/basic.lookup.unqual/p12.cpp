// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

struct S {};
S E0;

namespace {
  enum {
    E0 = 1,
    E1 = E0 + 1
  };
}


