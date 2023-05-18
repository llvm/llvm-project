// RUN: %clang_cc1 -fsyntax-only -Wunused -std=c++1z -verify %s

struct [[maybe_unused]] S {
  int I [[maybe_unused]];
  static int SI [[maybe_unused]];
};

enum [[maybe_unused]] E1 {
  EnumVal [[maybe_unused]]
};

[[maybe_unused]] void unused_func([[maybe_unused]] int parm) {
  typedef int maybe_unused_int [[maybe_unused]];
  [[maybe_unused]] int I;
}

namespace [[maybe_unused]] N {} // expected-warning {{'maybe_unused' attribute only applies to}}
