// RUN: %clang_analyze_cc1 -analyzer-output=text        \
// RUN:     -analyzer-checker=core -verify %s

int assuming_unsigned_ge_0(unsigned arg) {
  // expected-note@+2 {{'arg' is >= 0}}
  // expected-note@+1 {{Taking false branch}}
  if (arg < 0)
    return 0;
  // expected-note@+2 {{Assuming 'arg' is <= 0}}
  // expected-note@+1 {{Taking false branch}}
  if (arg > 0)
    return 0;
  // expected-note@+2 {{Division by zero}}
  // expected-warning@+1 {{Division by zero}}
  return 100 / arg;
}
