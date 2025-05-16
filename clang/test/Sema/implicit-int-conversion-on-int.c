// RUN: %clang_cc1 %s -verify -Wimplicit-int-conversion
// RUN: %clang_cc1 %s -verify -Wimplicit-int-conversion -Wno-implicit-int-conversion-on-negation -DNO_DIAG

char test_char(char x) {
  return -x;
#ifndef NO_DIAG
  // expected-warning@-2 {{implicit conversion loses integer precision}}
#else
  // expected-no-diagnostics
#endif
}

unsigned char test_unsigned_char(unsigned char x) {
  return -x; 
#ifndef NO_DIAG
  // expected-warning@-2 {{implicit conversion loses integer precision}}
#else
  // expected-no-diagnostics
#endif
}

short test_short(short x) {
  return -x; 
#ifndef NO_DIAG
  // expected-warning@-2 {{implicit conversion loses integer precision}}
#else
  // expected-no-diagnostics
#endif
}

unsigned short test_unsigned_short(unsigned short x) {
  return -x;
#ifndef NO_DIAG
  // expected-warning@-2 {{implicit conversion loses integer precision}}
#else
  // expected-no-diagnostics
#endif
}

// --- int-width and wider (should NOT warn) ---

int test_i(int x) {
  return -x; // no warning
}

unsigned int test_ui(unsigned int x) {
  return -x; // no warning
}

long test_l(long x) {
  return -x; // no warning
}

unsigned long test_ul(unsigned long x) {
  return -x; // no warning
}

long long test_ll(long long x) {
  return -x; // no warning
}

unsigned long long test_ull(unsigned long long x) {
  return -x; // no warning
}
