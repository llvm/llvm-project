// RUN: %clang_cc1 %s -verify=expected -Wimplicit-int-conversion
// RUN: %clang_cc1 %s -verify=none -Wimplicit-int-conversion -Wno-implicit-int-conversion-on-negation

// none-no-diagnostics

char test_char(char x) {
  return -x; // expected-warning {{implicit conversion loses integer precision: 'int' to 'char' on negation}}
}

unsigned char test_unsigned_char(unsigned char x) {
  return -x; // expected-warning {{implicit conversion loses integer precision: 'int' to 'unsigned char' on negation}}
}

short test_short(short x) {
  return -x; // expected-warning {{implicit conversion loses integer precision: 'int' to 'short' on negation}}
}

unsigned short test_unsigned_short(unsigned short x) {
  return -x; // expected-warning {{implicit conversion loses integer precision: 'int' to 'unsigned short' on negation}}
}

// --- int-width and wider (should NOT warn) ---

int test_i(int x) {
  return -x;
}

unsigned int test_ui(unsigned int x) {
  return -x;
}

long test_l(long x) {
  return -x;
}

unsigned long test_ul(unsigned long x) {
  return -x;
}

long long test_ll(long long x) {
  return -x;
}

unsigned long long test_ull(unsigned long long x) {
  return -x;
}

unsigned _BitInt(16) test_unsigned_bit_int(unsigned _BitInt(16) x) {
  return -x;
}

unsigned test_shift_minus(int i) {
  return -(1 << i);
}

unsigned test_shift_not(int i) {
  return ~(1 << i);
}
