// RUN: %clang_analyze_cc1 -analyze -analyzer-checker=core -verify %s

// expected-no-diagnostics
int **h;
int overflow_in_memregion(long j) {
  for (int l = 0;; ++l) {
    if (j - l > 0)
      return h[j - l][0]; // no-crash
  }
  return 0;
}
