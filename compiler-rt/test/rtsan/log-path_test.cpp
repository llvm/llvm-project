// RUN: %clangxx -fsanitize=realtime %s -o %t
// UNSUPPORTED: ios

// Regular run.
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

// Good log_path.
// RUN: rm -f %t.log.*
// RUN: %env_rtsan_opts=log_path=%t.log not %run %t 2> %t.out
// RUN: FileCheck %s --check-prefix=CHECK-ERROR < %t.log.*

#include <stdio.h>
#include <stdlib.h>

void violation() [[clang::nonblocking]] {
  void *ptr = malloc(2);
  printf("ptr: %p\n", ptr);
}

int main() {
  violation();
  return 0;
}

// CHECK-ERROR: ==ERROR: RealtimeSanitizer: unsafe-library-call
// CHECK-ERROR-NEXT: Intercepted call to real-time unsafe function `malloc` in real-time context!
// CHECK-ERROR: SUMMARY: RealtimeSanitizer: unsafe-library-call
