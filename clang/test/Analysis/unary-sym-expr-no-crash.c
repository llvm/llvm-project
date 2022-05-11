// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -analyzer-config support-symbolic-integer-casts=false \
// RUN:   -verify

// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -analyzer-config support-symbolic-integer-casts=true \
// RUN:   -verify

// expected-no-diagnostics

void clang_analyzer_eval(int);
void clang_analyzer_dump(int);

void crash(int b, long c) {
  b = c;
  if (b > 0)
    if(-b) // should not crash here
      ;
}
