// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=true \
// RUN:   -verify

// Here we test that no assertion is fired during symbol simplification.
// Related issue: https://github.com/llvm/llvm-project/issues/55546

extern void abort() __attribute__((__noreturn__));
#define assert(expr) ((expr) ? (void)(0) : abort())

void clang_analyzer_warnIfReached();

int a, b, c;
long L, L1;
void test() {
  assert(a + L + 1 + b != c);
  assert(L == a);
  assert(c == 0);
  L1 = 0;
  assert(a + L1 + 1 + b != c);
  assert(a == 0); // no-assertion
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}
