// REQUIRES: asserts
// RUN: %clang_analyze_cc1 -analyze -analyzer-checker=core -mllvm -debug %s 2>&1 | FileCheck %s

int **h;
int overflow_in_memregion(long j) {
  for (int l = 0;; ++l) {
    if (j - l > 0)
      return h[j - l][0]; // no-crash
  }
  return 0;
}
// CHECK: {{.*}}
// CHECK: MemRegion::getAsArrayOffset: offset overflowing, returning unknown
