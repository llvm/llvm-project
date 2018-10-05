// RUN: %clang_esan_wset -O0 %s -o %t 2>&1
// RUN: %run %t 2>&1 | FileCheck %s

// FIXME: Re-enable once PR33590 is fixed.
// UNSUPPORTED: x86_64
// Stucks at init and no clone feature equivalent.
// UNSUPPORTED: freebsd

#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <assert.h>

const int size = 0x1 << 25; // 523288 cache lines
const int line_size = 64;

int main(int argc, char **argv) {
  char *bufA = (char *)malloc(sizeof(char) * line_size);
  char bufB[64];
  char *bufC = (char *)mmap(0, size, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  bufA[0] = 0;
  // This additional access to the same line should not increase the line
  // count: but it's difficult to make a non-flaky test that measures the
  // lines down to the ones digit so right now we're not really testing that.
  // If we add a heap-only mode we may be able to be more precise.
  bufA[1] = 0;
  bufB[33] = 1;
  for (int i = 0; i < size; i += line_size)
    bufC[i] = 0;
  free(bufA);
  munmap(bufC, 0x4000);
  // CHECK: {{.*}} EfficiencySanitizer: the total working set size: 32 MB (524{{[0-9][0-9][0-9]}} cache lines)
  return 0;
}
