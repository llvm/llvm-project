// RUN: %clang_cc1 -triple=i386-apple-darwin10 -analyze -analyzer-checker=debug.ExprInspection -verify %s
void clang_analyzer_eval(int);

// pr28449: Used to crash.
void foo(void) {
  static const unsigned short array[] = (const unsigned short[]){0x0F00};
  // FIXME: Should be true.
  clang_analyzer_eval(array[0] == 0x0F00); // expected-warning{{UNKNOWN}}
}
