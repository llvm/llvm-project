// Function not intercepted on android.
// UNSUPPORTED: android

// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <stdlib.h>
#include <wchar.h>

int main() {
  char *buff = (char*) malloc(MB_CUR_MAX);
  free(buff);
  wcrtomb(buff, L'a', NULL);
  // CHECK: use-after-free
  // CHECK: SUMMARY
  return 0;
}
