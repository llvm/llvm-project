// RUN: %clang_analyze_cc1 -ffixed-point \
// RUN:   -analyzer-checker=core,debug.ExprInspection -verify %s

// expected-no-diagnostics

// Check that getAPSIntType does not crash
// when using fixed point types.

enum Kind { en_0 = 1 };

void _enum(int c) {
  (void)((enum Kind) c >> 4);
}

void _inttype(int c) {
  (void)(c >> 4);
}

void _accum(int c) {
  (void)((_Accum) c >> 4);
}

void _fract(int c) {
  (void)((_Fract) c >> 4);
}

void _long_fract(int c) {
  (void)((long _Fract) c >> 4);
}

void _unsigned_accum(int c) {
  (void)((unsigned _Accum) c >> 4);
}

void _short_unsigned_accum(int c) {
  (void)((short unsigned _Accum) c >> 4);
}

void _unsigned_fract(int c) {
  (void)((unsigned _Fract) c >> 4);
}

void sat_accum(int c) {
  (void)((_Sat _Accum) c >> 4);
}

void sat_fract(int c) {
  (void)((_Sat _Fract) c >> 4);
}

void sat_long_fract(int c) {
  (void)((_Sat long _Fract) c >> 4);
}

void sat_unsigned_accum(int c) {
  (void)((_Sat unsigned _Accum) c >> 4);
}

void sat_short_unsigned_accum(int c) {
  (void)((_Sat short unsigned _Accum) c >> 4);
}

void sat_unsigned_fract(int c) {
  (void)((_Sat unsigned _Fract) c >> 4);
}
