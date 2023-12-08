// RUN: %clang_analyze_cc1 -analyze -analyzer-checker=core -verify %s

int **h;
int overflow_in_memregion(long j) {
  for (int l = 0;; ++l) {
    if (j - l > 0)
      return h[j - l][0]; // no-crash
  }
  return 0;
}

void rdar39593879(long long *d) {
  long e, f;
  e = f = d[1]; // no-crash
  for (; d[e];) f-- > 0; // expected-warning{{relational comparison result unused}}; 
}
