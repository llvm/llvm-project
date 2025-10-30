// RUN: %clangxx_asan -O0 %s -o %t && not %env_asan_opts=log_to_stderr=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK
// RUN: %clangxx_asan -O1 %s -o %t && not %env_asan_opts=log_to_stderr=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK
// RUN: %clangxx_asan -O2 %s -o %t && not %env_asan_opts=log_to_stderr=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK
// RUN: %clangxx_asan -O3 %s -o %t && not %env_asan_opts=log_to_stderr=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK

#include <stdio.h>
#include <wchar.h>

int main() {
  const wchar_t *src = L"X means dog";
  wchar_t goodDst[12];
  wcsncpy(goodDst, src, 12);

  wchar_t badDst[7];
  wcsncpy(badDst, src, 7); // This should still work.
  fprintf(stderr, "Good so far.\n");
  // CHECK-DAG: Good so far.
  fflush(stderr);
  wcsncpy(badDst, src, 15); // Boom!
  // CHECK-DAG: ERROR: AddressSanitizer: stack-buffer-overflow on address [[ADDR:0x[0-9a-f]+]] at pc {{0x[0-9a-f]+}} bp {{0x[0-9a-f]+}} sp {{0x[0-9a-f]+}}
  // CHECK-DAG: WRITE of size {{[0-9]+}} at [[ADDR]] thread T0
  // CHECK-DAG: #0 {{0x[0-9a-f]+}} in wcsncpy
  printf("Should have failed with ASAN error.\n");
}
