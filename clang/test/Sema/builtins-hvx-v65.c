// REQUIRES: hexagon-registered-target
// RUN: %clang_cc1 -triple hexagon %s -target-feature +hvx-length128b -target-feature +hvxv65 -target-cpu hexagonv65 -fsyntax-only -verify

typedef long Vect1024 __attribute__((__vector_size__(128)))
    __attribute__((aligned(128)));
typedef long Vect2048 __attribute__((__vector_size__(256)))
    __attribute__((aligned(128)));

typedef Vect1024 HVX_Vector;
typedef Vect2048 HVX_VectorPair;

// expected-no-diagnostics
HVX_Vector builtin_needs_v60(HVX_VectorPair a) {
  return __builtin_HEXAGON_V6_hi_128B(a);
}

HVX_Vector builtin_needs_v62(char a) {
  return __builtin_HEXAGON_V6_lvsplatb_128B(a);
}

HVX_VectorPair builtin_needs_v65() {
  return __builtin_HEXAGON_V6_vdd0_128B();
}
