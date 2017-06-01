// Check that ASan dumps the CPU registers on a SIGSEGV.

// RUN: %clangxx_asan %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <stdio.h>

int main() {
  fprintf(stderr, "Hello\n");
  char *ptr;

  if (sizeof(void *) == 8)
    ptr = (char *)0x6666666666666666;
  else if (sizeof(void *) == 4)
    ptr = (char *)0x55555555;
  else
    assert(0 && "Your computer is weird.");

  char c = *ptr;  // BOOM
  // CHECK: ERROR: AddressSanitizer: SEGV
  // CHECK: Register values:
  // CHECK: {{0x55555555|0x6666666666666666}}
  fprintf(stderr, "World\n");
  return c;
}
