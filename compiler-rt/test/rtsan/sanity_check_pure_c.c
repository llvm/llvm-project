// RUN: %clang -fsanitize=realtime %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: %clang %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NO-SANITIZE
#ifdef __cplusplus
#  error "This test must be built in C mode"
#endif

#include <stdio.h>
#include <stdlib.h>

// Check that we can build and run C code.

void nonblocking_function(void) __attribute__((nonblocking));

void nonblocking_function(void) __attribute__((nonblocking)) {
  void *ptr = malloc(2);
  printf("ptr: %p\n", ptr); // ensure we don't optimize out the malloc
}

int main() {
  nonblocking_function();
  printf("Done\n");
  return 0;
}

// CHECK: ==ERROR: RealtimeSanitizer
// CHECK-NO-SANITIZE: Done
