// RUN: %clang_asan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

int main(int argc, const char * argv[]) {
  getpwnam(NULL);
  fprintf(stderr, "Finished.\n");
  return 0;
}

// CHECK: Finished.
