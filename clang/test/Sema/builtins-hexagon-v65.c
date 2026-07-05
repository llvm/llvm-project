// REQUIRES: hexagon-registered-target
// RUN: %clang_cc1 %s -triple hexagon -fsyntax-only -verify -target-cpu hexagonv65

// expected-no-diagnostics
unsigned builtin_needs_v60(unsigned Rs) {
  return __builtin_HEXAGON_S6_rol_i_r(Rs, 3);
}

unsigned long long builtin_needs_v62(unsigned Rs) {
  return __builtin_HEXAGON_S6_vsplatrbp(Rs);
}

unsigned builtin_needs_v65(unsigned long long Rss, unsigned long long Rtt) {
  return __builtin_HEXAGON_A6_vcmpbeq_notany(Rss, Rtt);
}
