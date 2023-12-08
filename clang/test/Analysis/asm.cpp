// RUN: %clang_analyze_cc1 -analyzer-checker debug.ExprInspection -fheinous-gnu-extensions -w %s -verify

int clang_analyzer_eval(int);

int global;
void testRValueOutput() {
  int &ref = global;
  ref = 1;
  __asm__("" : "=r"(((int)(global))));  // don't crash on rvalue output operand
  clang_analyzer_eval(global == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(ref == 1);    // expected-warning{{UNKNOWN}}
}
