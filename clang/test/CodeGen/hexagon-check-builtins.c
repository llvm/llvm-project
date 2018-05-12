// REQUIRES: hexagon-registered-target
// RUN: %clang_cc1 -fsyntax-only -triple hexagon-unknown-elf -verify %s

int foo(int x) {
  // expected-error@+2 {{argument should be a value from 0 to 31}}
  // expected-error@+1 {{argument should be a value from 0 to 31}}
  return __builtin_HEXAGON_S4_extract(x, 33, -1) +
  // expected-error@+1 {{argument should be a value from 0 to 31}}
         __builtin_HEXAGON_S4_extract(x, 3, 91) +
  // expected-error@+2 {{argument should be a value from 0 to 31}}
  // expected-error@+1 {{argument should be a value from 0 to 31}}
         __builtin_HEXAGON_S4_extract(x, -1, 35) +
         __builtin_HEXAGON_S4_extract(x, 0, 31) +
         __builtin_HEXAGON_S4_extract(x, 31, 0);
}

int bar(void *p, void *q, int x) {
  // expected-error@+1 {{argument should be a multiple of 4}}
  return __builtin_HEXAGON_L2_loadri_pci(p, -1, x, q) +
  // expected-error@+2 {{argument should be a value from -32 to 28}}
  // expected-error@+1 {{argument should be a multiple of 4}}
         __builtin_HEXAGON_L2_loadri_pci(p, -99, x, q) +
  // expected-error@+1 {{argument should be a value from -32 to 28}}
         __builtin_HEXAGON_L2_loadri_pci(p, -132, x, q) +
         __builtin_HEXAGON_L2_loadri_pci(p, 28, x, q) +
  // expected-error@+2 {{argument should be a value from -32 to 28}}
  // expected-error@+1 {{argument should be a multiple of 4}}
         __builtin_HEXAGON_L2_loadri_pci(p, 29, x, q);
}

