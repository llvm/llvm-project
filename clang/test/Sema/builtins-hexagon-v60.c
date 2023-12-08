// REQUIRES: hexagon-registered-target
// RUN: %clang_cc1 %s -triple hexagon -verify -target-cpu hexagonv60 -DTEST_V60 -S -o -
// RUN: %clang_cc1 %s -triple hexagon -verify -target-cpu hexagonv60 -DTEST_V62 -S -o -
// RUN: %clang_cc1 %s -triple hexagon -verify -target-cpu hexagonv60 -DTEST_V65 -S -o -

#ifdef TEST_V60
unsigned builtin_needs_v60(unsigned Rs) {
  // expected-no-diagnostics
  return __builtin_HEXAGON_S6_rol_i_r(Rs, 3);
}
#endif

#ifdef TEST_V62
unsigned long long builtin_needs_v62(unsigned Rs) {
  // expected-error-re@+1 {{'__builtin_HEXAGON_S6_vsplatrbp' needs target feature v62|{{.*}}}}
  return __builtin_HEXAGON_S6_vsplatrbp(Rs);
}
#endif

#ifdef TEST_V65
unsigned builtin_needs_v65(unsigned long long Rss, unsigned long long Rtt) {
  // expected-error-re@+1 {{'__builtin_HEXAGON_A6_vcmpbeq_notany' needs target feature v65|{{.*}}}}
  return __builtin_HEXAGON_A6_vcmpbeq_notany(Rss, Rtt);
}
#endif
