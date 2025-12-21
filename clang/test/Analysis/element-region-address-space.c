// RUN: %clang_analyze_cc1 -triple amdgcn-unknown-unknown \
// RUN:   -Wno-incompatible-library-redeclaration \
// RUN:   -analyzer-checker=core,unix -verify %s

// expected-no-diagnostics
//
// By default, pointers are 64-bits.
#define ADDRESS_SPACE_64BITS __attribute__((address_space(0)))
#define ADDRESS_SPACE_32BITS __attribute__((address_space(3)))

int test(ADDRESS_SPACE_32BITS int *p, ADDRESS_SPACE_32BITS void *q) {
  return p == q; // no-crash
}

// Make sure that the cstring checker handles non-default address spaces
ADDRESS_SPACE_64BITS void *
memcpy(ADDRESS_SPACE_64BITS void *,
       ADDRESS_SPACE_32BITS const void *,
       long unsigned int);

ADDRESS_SPACE_64BITS struct {
  char m[16];
} n;

void avoid_cstring_checker_crash(ADDRESS_SPACE_32BITS char *p) {
  memcpy(&n.m[0], p, 4); // no-crash
}
