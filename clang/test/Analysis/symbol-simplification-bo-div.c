// RUN: %clang_analyze_cc1 -analyzer-checker=core %s \
// RUN:    -triple x86_64-pc-linux-gnu -verify

// don't crash
// expected-no-diagnostics

int a, b;
int c(void) {
  unsigned d = a;
  --d;
  short e = b / b - a;
  ++e;
  return d <= 0 && e && e;
}
