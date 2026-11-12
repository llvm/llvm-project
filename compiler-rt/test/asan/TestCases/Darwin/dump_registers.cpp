// Check that ASan dumps the CPU registers on a SIGSEGV.

// RUN: %clangxx_asan %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <stdio.h>
#include <sys/mman.h>

int main() {
  fprintf(stderr, "Hello\n");
  char *ptr;

  ptr = (char *)mmap(NULL, 0x10000, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0);
  assert(ptr && "failed to mmap");

  fprintf(stderr, sizeof(uintptr_t) == 8 ? "p = 0x%016lx\n" : "p = 0x%08lx\n", (uintptr_t)ptr);
  // CHECK: p = [[ADDR:0x[0-9]+]]

  char c = *ptr;  // BOOM
  // CHECK: ERROR: AddressSanitizer: {{SEGV|BUS}}
  // CHECK: Register values:
  // CHECK: [[ADDR]]
  fprintf(stderr, "World\n");
  return c;
}
