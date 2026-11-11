// REQUIRES: hexagon-registered-target
// RUN: %clang_cc1 -fsyntax-only -triple hexagon -target-cpu hexagonv85 \
// RUN:   -target-feature +hlx -verify %s

// There is no HLX vector element extract, so subscripts are rejected.

typedef short HLX_Vector __attribute__((__vector_size__(2048))) __attribute__((aligned(2048)));
typedef long HLX_VectorPair __attribute__((__vector_size__(4096))) __attribute__((aligned(2048)));

long f(HLX_Vector v, HLX_VectorPair vp, int i) {
  HLX_Vector arr[2][3];
  arr[0][1] = v;
  long val = vp[1]; // expected-error{{subscript of HLX vector type is not allowed}}
  val += vp[i];     // expected-error{{subscript of HLX vector type is not allowed}}
  val += v[2];      // expected-error{{subscript of HLX vector type is not allowed}}
  val += arr[0][1][3]; // expected-error{{subscript of HLX vector type is not allowed}}
  return val;
}
