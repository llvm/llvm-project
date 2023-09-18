 // RUN: %clang_analyze_cc1 -analyzer-checker=core \
 // RUN:   -analyzer-checker=debug.ExprInspection \
 // RUN:   -verify %s

// Don't crash when using _BitInt()
// expected-no-diagnostics
_BitInt(256) a;
_BitInt(129) b;
void c() {
  b = a;
}
