// RUN: %clang_cc1 -fsyntax-only -verify -Wall %s
// expected-no-diagnostics
void** f(void **Buckets, unsigned NumBuckets) {
  return Buckets + NumBuckets;
}
