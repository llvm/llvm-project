// RUN: %clang_analyze_cc1 -triple amdgcn-unknown-unknown \
// RUN:   -analyzer-checker=core -verify %s

// expected-no-diagnostics
//
// By default, pointers are 64-bits.
#define ADDRESS_SPACE_32BITS __attribute__((address_space(3)))

int test(ADDRESS_SPACE_32BITS int *p, ADDRESS_SPACE_32BITS void *q) {
  return p == q; // no-crash
}
