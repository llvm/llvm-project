// RUN: %clang_cc1 -analyze -analyzer-checker=core -verify %s
// expected-no-diagnostics

// Test when entering f1(), we set the right AnalysisDeclContext to Environment.
// Otherwise, block-level expr '1 && a' would not be block-level.
int a;

void f1() {
  if (1 && a)
    return;
}

void f2() {
  f1();
}
