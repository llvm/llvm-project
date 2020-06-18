// RUN: %clangxx_msan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: %clangxx_msan -O1 %s -o %t && %run %t >%t.out 2>&1
// RUN: %clangxx_msan -O2 %s -o %t && %run %t >%t.out 2>&1
// RUN: %clangxx_msan -O3 %s -o %t && %run %t >%t.out 2>&1

// Test that (no_sanitize_memory) functions DO NOT propagate shadow.

#include <stdlib.h>
#include <stdio.h>

__attribute__((noinline))
__attribute__((weak))
__attribute__((no_sanitize_memory))
int f(int x) {
  return x;
}

int main(void) {
  int x;
  int * volatile p = &x;
  int y = f(*p);
  if (y)
    exit(0);
  return 0;
}
