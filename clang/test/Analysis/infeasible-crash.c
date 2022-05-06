// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=alpha.unix.cstring.OutOfBounds,alpha.unix.cstring.UninitializedRead \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -verify

// expected-no-diagnostics

void *memmove(void *, const void *, unsigned long);

typedef struct {
  char a[1024];
} b;
int c;
b *invalidate();
int d() {
  b *a = invalidate();
  if (c < 1024)
    return 0;
  int f = c & ~3, g = f;
  g--;
  if (g)
    return 0;

  // Parent state is already infeasible.
  // clang_analyzer_printState();
  // "constraints": [
  //   { "symbol": "(derived_$3{conj_$0{int, LC1, S728, #1},c}) & -4", "range": "{ [1, 1] }" },
  //   { "symbol": "derived_$3{conj_$0{int, LC1, S728, #1},c}", "range": "{ [1024, 2147483647] }" }
  // ],

  // This sould not crash!
  // It crashes in baseline, since there both true and false states are nullptr!
  memmove(a->a, &a->a[f], c - f);

  return 0;
}
