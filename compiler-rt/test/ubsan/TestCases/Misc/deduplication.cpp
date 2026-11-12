// RUN: %clangxx -fsanitize=undefined %s -o %t && %run %t 2>&1 | FileCheck %s
// Verify deduplication works by ensuring only one diag is emitted.
#include <limits.h>
#include <stdio.h>

void overflow() {
  int i = INT_MIN;
  --i;
}

int main() {
  // CHECK: Start
  fprintf(stderr, "Start\n");
  fflush(stderr);

  // CHECK: runtime error
  // CHECK-NOT: runtime error
  // CHECK-NOT: runtime error
  overflow();
  overflow();
  overflow();

  // CHECK: End
  fprintf(stderr, "End\n");
  return 0;
}
