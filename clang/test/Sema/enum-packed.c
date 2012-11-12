// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// PR7477
enum __attribute__((packed)) E {
  Ea, Eb, Ec, Ed
};

void test_E(enum E e) {
  switch (e) {
  case Ea:
  case Eb:
  case Ec:
  case Ed:
    break;
  }
}
