// REQUIRES: !windows
// RUN: %clangxx_asan -O0 -fno-builtin %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <wchar.h>

__attribute__((noinline)) void bad_wcs(void) {
  wchar_t buf[] = L"hello";
  // CHECK: wcscpy-param-overlap: memory ranges
  wcscpy(buf, buf + 1);
}

int main() {
  bad_wcs();
  return 0;
}


