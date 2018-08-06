// REQUIRES: hexagon-registered-target
// RUN: %clang_cc1 %s -triple hexagon -fsyntax-only -verify -target-cpu hexagonv60

unsigned builtin_needs_v60(unsigned Rs) {
  return __builtin_HEXAGON_S6_rol_i_r(Rs, 3);
}

unsigned long long builtin_needs_v62(unsigned Rs) {
  // expected-error@+1 {{builtin is not supported on this CPU}}
  return __builtin_HEXAGON_S6_vsplatrbp(Rs);
}

unsigned builtin_needs_v65(unsigned long long Rss, unsigned long long Rtt) {
  // expected-error@+1 {{builtin is not supported on this CPU}}
  return __builtin_HEXAGON_A6_vcmpbeq_notany(Rss, Rtt);
}
