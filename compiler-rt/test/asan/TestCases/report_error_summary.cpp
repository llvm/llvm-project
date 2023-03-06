// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <stdio.h>

extern "C" void __sanitizer_report_error_summary(const char *summary) {
  fprintf(stderr, "test_report_error_summary\n", summary);
  // CHECK: test_report_error_summary
  fflush(stderr);
}

char *x;

int main() {
  x = new char[20];
  delete[] x;
  return x[0];
}
