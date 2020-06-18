// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s

// rdar://problem/44978988
// expected-no-diagnostics

int foo();

int gTotal;

double bar(int start, int end) {
  int i, cnt, processed, size;
  double result, inc;

  result = 0;
  processed = start;
  size = gTotal * 2;
  cnt = (end - start + 1) * size;

  for (i = 0; i < cnt; i += 2) {
    if ((i % size) == 0) {
      inc = foo();
      processed++;
    }
    result += inc * inc; // no-warning
  }
  return result;
}
