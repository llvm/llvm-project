// RUN: %clangxx -fsanitize=realtime %s -o %t
// RUN: env RTSAN_OPTIONS="halt_on_error=false" %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: ios

// Intent: Ensure that halt_on_error does not exit on the first violation.

#include <stdlib.h>

void *MallocViolation() { return malloc(10); }

void FreeViolation(void *Ptr) { free(Ptr); }

void process() [[clang::nonblocking]] {
  void *Ptr = MallocViolation();
  FreeViolation(Ptr);
}

int main() {
  process();
  return 0;
  // CHECK: ==ERROR: RealtimeSanitizer
  // CHECK-NEXT: {{.*`malloc`.*}}
  // CHECK: ==ERROR: RealtimeSanitizer
  // CHECK-NEXT: {{.*`free`.*}}
}
