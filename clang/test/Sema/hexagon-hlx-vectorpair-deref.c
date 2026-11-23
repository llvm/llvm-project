// REQUIRES: hexagon-registered-target
// RUN: %clang_cc1 -fsyntax-only -triple hexagon -target-cpu hexagonv85 \
// RUN:   -target-feature +hlx -verify %s

// There are no HLX vector pair loads or stores, so dereferences of vector
// pair pointers are rejected.

typedef short HLX_Vector __attribute__((__vector_size__(2048))) __attribute__((aligned(2048)));
typedef long HLX_VectorPair __attribute__((__vector_size__(4096))) __attribute__((aligned(2048)));

void f(HLX_Vector *src0, HLX_VectorPair *src1, HLX_VectorPair *dst) {
  HLX_VectorPair arr[2][3];
  HLX_VectorPair (*p)[2][3] = &arr;
  HLX_VectorPair *first = (*p)[1];
  HLX_Vector v = *src0;
  HLX_VectorPair a = *src1; // expected-error{{indirection not permitted on HLX vector pair pointer 'HLX_VectorPair *'}}
  HLX_VectorPair b = *first; // expected-error{{indirection not permitted on HLX vector pair pointer 'HLX_VectorPair *'}}
  *dst = a; // expected-error{{indirection not permitted on HLX vector pair pointer 'HLX_VectorPair *'}}
}
