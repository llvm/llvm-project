// RUN: %clang_cc1 -Wall -ffreestanding -fsyntax-only -fwrapv -verify %s

int test() {
  int i;
  i = -1 << 1; // no-warning
  return i;
}

// expected-no-diagnostics
