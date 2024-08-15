//RUN: %clang_analyze_cc1 -triple amdgcn-unknown-unknown -analyzer-checker=core -verify %s
// expected-no-diagnostics

#include <stdint.h>

void bar(__global int *p) __attribute__((nonnull(1)));

void foo(__global int *p) {
  if ((uint64_t)p <= 1UL << 32)
    bar(p); // no-warning
}
