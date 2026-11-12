 // RUN: %clang_analyze_cc1 -analyzer-checker=core \
 // RUN:   -analyzer-checker=debug.ExprInspection \
 // RUN:   -triple x86_64-pc-linux-gnu \
 // RUN:   -verify %s

// Don't crash when using _BitInt(). Pin to the x86_64 triple for now,
// since not all architectures support _BitInt()
// expected-no-diagnostics
_BitInt(256) a;
_BitInt(129) b;
void c() {
  b = a;
}
