// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run %t 0 2>&1 | FileCheck %s

#include <sanitizer/asan_interface.h>
#include <stdio.h>

static void ErrorReportCallbackOneToZ(const char *report) {
  fprintf(stderr, "ABCDEF%sGHIJKL", report);
  fflush(stderr);
}

int main(int argc, char **argv) {
  __asan_set_error_report_callback(ErrorReportCallbackOneToZ);
  __asan_report_error(
      (void *)__builtin_extract_return_addr(__builtin_return_address(0)), 0, 0,
      0, true, 1);
  // CHECK: ABCDEF
  // CHECK: ERROR: AddressSanitizer
  // CHECK: GHIJKL
  return 0;
}
