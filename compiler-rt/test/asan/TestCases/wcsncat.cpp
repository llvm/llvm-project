// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK
// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK
// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK
// RUN: %clangxx_asan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK

#include <stdio.h>
#include <wchar.h>

int main() {
  wchar_t *start = L"X means ";
  wchar_t *append = L"dog";
  wchar_t goodDst[15];
  wcscpy(goodDst, start);
  wcsncat(goodDst, append, 5);

  wchar_t badDst[11];
  wcscpy(badDst, start);
  wcsncat(badDst, append, 1);
  printf("Good so far.\n");
  // CHECK: Good so far.
  fflush(stdout);
  wcsncat(badDst, append, 3); // Boom!
  // CHECK: ERROR: AddressSanitizer: stack-buffer-overflow on address [[ADDR:0x[0-9a-f]+]] at pc {{0x[0-9a-f]+}} bp {{0x[0-9a-f]+}} sp {{0x[0-9a-f]+}}
  // CHECK: WRITE of size {{[0-9]+}} at [[ADDR:0x[0-9a-f]+]] thread T0
  // CHECK: #0 [[ADDR:0x[0-9a-f]+]] in wcsncat{{.*}}sanitizer_common_interceptors.inc:{{[0-9]+}}
  printf("Should have failed with ASAN error.\n");
}