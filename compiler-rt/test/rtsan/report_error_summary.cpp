// RUN: %clangxx -fsanitize=realtime %s -o %t
// RUN: %env_rtsan_opts="halt_on_error=false" %run %t 2>&1 | FileCheck %s

// RUN: %clangxx -DTEST_CUSTOM_HANDLER=1 -fsanitize=realtime %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK-CUSTOM-HANDLER

// UNSUPPORTED: ios

// Intent: Make sure we support ReporErrorSummary, including custom handlers

#include <stdio.h>
#include <stdlib.h>

#ifdef TEST_CUSTOM_HANDLER
extern "C" void __sanitizer_report_error_summary(const char *error_summary) {
  fprintf(stderr, "%s %s\n", "In custom handler! ", error_summary);
}
#endif

int blocking_call() [[clang::blocking]] { return 0; }

int main() [[clang::nonblocking]] {
  void *ptr = malloc(2);
  blocking_call();

  printf("ptr: %p\n", ptr); // ensure we don't optimize out the malloc
}

// CHECK: SUMMARY: RealtimeSanitizer: unsafe-library-call
// CHECK: SUMMARY: RealtimeSanitizer: blocking-call

// CHECK-CUSTOM-HANDLER: In custom handler! SUMMARY: RealtimeSanitizer: unsafe-library-call
