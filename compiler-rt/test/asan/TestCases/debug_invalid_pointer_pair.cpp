// Checks that the ASan debugging API for getting report information
// returns correct values for invalid pointer pairs.
// RUN: %clangxx_asan -O0 %s -o %t -mllvm -asan-detect-invalid-pointer-pair && %env_asan_opts=detect_invalid_pointer_pairs=1 not %run %t 2>&1 | FileCheck %s

#include <sanitizer/asan_interface.h>
#include <stdio.h>
#include <stdlib.h>

char *p;
char *q;

int main() {
  // Disable stderr buffering. Needed on Windows.
  setvbuf(stderr, NULL, _IONBF, 0);

  p = (char *)malloc(42);
  q = (char *)malloc(42);

  fprintf(stderr, "p: %p\n", p);
  // CHECK: p: 0x[[ADDR1:[0-9a-f]+]]
  fprintf(stderr, "q: %p\n", q);
  // CHECK: q: 0x[[ADDR2:[0-9a-f]+]]

  // Trigger invalid pointer pair
  int res = p > q; // BOOM

  free(p);
  free(q);
  return res;
}

// Required for dyld macOS 12.0+
#if (__APPLE__)
__attribute__((weak))
#endif
extern "C" void
__asan_on_error() {
  int present = __asan_report_present();
  fprintf(stderr, "%s\n", (present == 1) ? "report" : "");
  // CHECK: report

  void *addr_first = NULL;
  size_t size_first = 0;
  int is_first = __asan_get_report_address_info(__asan_address_info_first, &addr_first, &size_first);
  fprintf(stderr, "is_first: %d, addr_first: %p, size_first: %ld\n", is_first, addr_first, size_first);
  // CHECK: is_first: 1, addr_first: 0x[[ADDR1]], size_first: 0

  void *addr_second = NULL;
  size_t size_second = 0;
  int is_second = __asan_get_report_address_info(__asan_address_info_second, &addr_second, &size_second);
  fprintf(stderr, "is_second: %d, addr_second: %p, size_second: %ld\n", is_second, addr_second, size_second);
  // CHECK: is_second: 1, addr_second: 0x[[ADDR2]], size_second: 0
}

// CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
