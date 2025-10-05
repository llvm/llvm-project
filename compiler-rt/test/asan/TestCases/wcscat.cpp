// RUN: %clangxx_asan -O0 %s -o %t && not %env_asan_opts=log_to_stderr=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK
// RUN: %clangxx_asan -O1 %s -o %t && not %env_asan_opts=log_to_stderr=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK
// RUN: %clangxx_asan -O2 %s -o %t && not %env_asan_opts=log_to_stderr=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK
// RUN: %clangxx_asan -O3 %s -o %t && not %env_asan_opts=log_to_stderr=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK

#include <stdio.h>
#include <wchar.h>

int main() {
  const wchar_t *start = L"X means ";
  const wchar_t *append = L"dog";
  wchar_t goodDst[12];
  wcscpy(goodDst, start);
  wcscat(goodDst, append);

  wchar_t badDst[9];
  wcscpy(badDst, start);
  fprintf(stderr, "Good so far.\n");
  // CHECK: Good so far.
  fflush(stderr);
  wcscat(badDst, append); // Boom!
  // CHECK: ERROR: AddressSanitizer: stack-buffer-overflow on address [[ADDR:0x[0-9a-f]+]] at pc {{0x[0-9a-f]+}} bp {{0x[0-9a-f]+}} sp {{0x[0-9a-f]+}}
  // CHECK: WRITE of size {{[0-9]+}} at [[ADDR]] thread T0
  // CHECK: #0 {{0x[0-9a-f]+}} in wcscat
  printf("Should have failed with ASAN error.\n");
}