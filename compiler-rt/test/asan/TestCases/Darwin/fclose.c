// RUN: %clang_asan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

int main(int argc, const char * argv[]) {
  fclose(NULL);
  fprintf(stderr, "Finished.\n");
  return 0;
}

// CHECK: Finished.
