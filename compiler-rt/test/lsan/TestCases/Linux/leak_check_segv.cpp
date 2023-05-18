// Test that SIGSEGV during leak checking does not crash the process.
// RUN: %clangxx_lsan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: ppc
#include <sanitizer/lsan_interface.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

char data[10 * 1024 * 1024];

int main() {
  long pagesize_mask = sysconf(_SC_PAGESIZE) - 1;
  void *p = malloc(10 * 1024 * 1024);
  // surprise-surprise!
  mprotect((void *)(((unsigned long)p + pagesize_mask) & ~pagesize_mask),
           16 * 1024, PROT_NONE);
  mprotect((void *)(((unsigned long)data + pagesize_mask) & ~pagesize_mask),
           16 * 1024, PROT_NONE);
  __lsan_do_leak_check();
  fprintf(stderr, "DONE\n");
}

// CHECK: Tracer caught signal 11
// CHECK: LeakSanitizer has encountered a fatal error
// CHECK: HINT: For debugging, try setting {{.*}} LSAN_OPTIONS
// CHECK-NOT: DONE
