// RUN: %clangxx -fsanitize=realtime %s -o %t
// RUN: %env_rtsan_opts="halt_on_error=true" not %run %t 2>&1 | FileCheck %s
// RUN: %env_rtsan_opts="halt_on_error=false" %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK-NO-HALT,CHECK
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
  // CHECK-NO-HALT: ==ERROR: RealtimeSanitizer
  // CHECK-NO-HALT-NEXT: {{.*`free`.*}}
}
