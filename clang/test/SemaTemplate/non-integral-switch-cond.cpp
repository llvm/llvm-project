// RUN: %clang_cc1 -fsyntax-only -verify %s

struct NOT_AN_INTEGRAL_TYPE {};

template <typename T>
struct foo {
  NOT_AN_INTEGRAL_TYPE Bad;
  void run() {
    switch (Bad) { // expected-error {{statement requires expression of integer type ('NOT_AN_INTEGRAL_TYPE' invalid)}}
    case 0:
      break;
    }
  }
};
