// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

int n;
const int N = 10000;

static void atexit1() {
  n++;
}

static void atexit0() {
  fprintf(stderr, "run count: %d\n", n);
}

int main() {
  atexit(atexit0);
  for (int i = 0; i < N; i++)
    atexit(atexit1);
}

// CHECK-NOT: FATAL: ThreadSanitizer
// CHECK-NOT: WARNING: ThreadSanitizer
// CHECK: run count: 10000

