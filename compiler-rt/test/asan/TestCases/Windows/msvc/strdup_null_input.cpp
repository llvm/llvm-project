// RUN: %clang_cl_asan %s -Fe%t.exe
// RUN: %run %t.exe | FileCheck %s

// CHECK: Success

#include <malloc.h>
#include <stdio.h>
#include <string.h>

int main() {
  // Null input is valid to strdup on Windows.
  char *nullStr = _strdup(nullptr);
  free(nullStr);
  puts("Success");
  return 0;
}
