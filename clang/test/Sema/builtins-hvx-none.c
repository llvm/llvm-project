// REQUIRES: hexagon-registered-target
// RUN: %clang_cc1 -triple hexagon %s -target-cpu hexagonv60 -DTEST_HVXV60 -verify -S -o -
// RUN: %clang_cc1 -triple hexagon %s -target-cpu hexagonv60 -DTEST_HVXV62 -verify -S -o -
// RUN: %clang_cc1 -triple hexagon %s -target-cpu hexagonv60 -DTEST_HVXV65 -verify -S -o -

typedef long Vect1024 __attribute__((__vector_size__(128)))
    __attribute__((aligned(128)));
typedef long Vect2048 __attribute__((__vector_size__(256)))
    __attribute__((aligned(128)));

typedef Vect1024 HVX_Vector;
typedef Vect2048 HVX_VectorPair;

#ifdef TEST_HVXV60
HVX_Vector builtin_needs_v60(HVX_VectorPair a) {
  // expected-error-re@+1 {{'__builtin_HEXAGON_V6_hi_128B' needs target feature hvxv60|{{.*}}}}
  return __builtin_HEXAGON_V6_hi_128B(a);
}
#endif

#ifdef TEST_HVXV62
HVX_Vector builtin_needs_v62(char a) {
  // expected-error-re@+1 {{'__builtin_HEXAGON_V6_lvsplatb_128B' needs target feature hvxv62|{{.*}}}}
  return __builtin_HEXAGON_V6_lvsplatb_128B(a);
}
#endif

#ifdef TEST_HVXV65
HVX_VectorPair builtin_needs_v65() {
  // expected-error-re@+1 {{'__builtin_HEXAGON_V6_vdd0_128B' needs target feature hvxv65|{{.*}}}}
  return __builtin_HEXAGON_V6_vdd0_128B();
}
#endif
