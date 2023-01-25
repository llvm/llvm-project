// Test that SIGSEGV during leak checking does not crash the process.
// RUN: %clangxx_lsan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
#include <stdlib.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sanitizer/lsan_interface.h>

char data[10 * 1024 * 1024];

int main() {
  void *p = malloc(10 * 1024 * 1024);
  // surprise-surprise!
  mprotect((void*)(((unsigned long)p + 4095) & ~4095), 16 * 1024, PROT_NONE);
  mprotect((void*)(((unsigned long)data + 4095) & ~4095), 16 * 1024, PROT_NONE);
  __lsan_do_leak_check();
  fprintf(stderr, "DONE\n");
}

// CHECK: Tracer caught signal 11
// CHECK: LeakSanitizer has encountered a fatal error
// CHECK: HINT: For debugging, try setting {{.*}} LSAN_OPTIONS
// CHECK-NOT: DONE
