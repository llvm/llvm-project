// RUN: %clang_cc1 -fsyntax-only -verify -Wall %s
// expected-no-diagnostics
// rdar://5683899
void** f(void **Buckets, unsigned NumBuckets) {
  return Buckets + NumBuckets;
}
