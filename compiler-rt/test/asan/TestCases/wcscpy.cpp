// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK
// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK
// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK
// RUN: %clangxx_asan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK

#include <stdio.h>
#include <wchar.h>

int main() {
  wchar_t *src = L"X means dog";
  wchar_t goodDst[12];
  wcscpy(goodDst, src);

  wchar_t badDst[7];
  printf("Good so far.\n");
  // CHECK: Good so far.
  fflush(stdout);
  wcscpy(badDst, src); // Boom!
  // CHECK:ERROR: AddressSanitizer: stack-buffer-overflow on address [[ADDR:0x[0-9a-f]+]] at pc {{0x[0-9a-f]+}} bp {{0x[0-9a-f]+}} sp {{0x[0-9a-f]+}}
  // CHECK: WRITE of size {{[0-9]+}} at [[ADDR:0x[0-9a-f]+]] thread T0
  // CHECK: #0 [[ADDR:0x[0-9a-f]+]] in wcscpy{{.*}}asan_interceptors.cpp:{{[0-9]+}}
  printf("Should have failed with ASAN error.\n");
}