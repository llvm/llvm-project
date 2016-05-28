// RUN: %clang_esan_wset -O0 %s -o %t 2>&1
// RUN: %run %t 2>&1 | FileCheck %s

#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <assert.h>
#include <string.h>

int main(int argc, char **argv) {
  const int size = 128*1024*1024;
  char *p = (char *)mmap(0, size, PROT_READ | PROT_WRITE,
                         MAP_ANON | MAP_PRIVATE, -1, 0);
  // Test the slowpath at different cache line boundaries.
  for (int i = 0; i < 630; i++)
    memset((char *)p + 63*i, i, 63*i);
  munmap(p, size);
  return 0;
  // FIXME: once the memory scan and size report is in place add it here.
  // CHECK: {{.*}}EfficiencySanitizer is not finished: nothing yet to report
}
