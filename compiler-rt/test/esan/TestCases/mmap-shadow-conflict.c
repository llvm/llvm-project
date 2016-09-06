// RUN: %clang_esan_frag -O0 %s -o %t 2>&1
// RUN: %env_esan_opts=verbosity=1 %run %t 2>&1 | FileCheck %s

#include <unistd.h>
#include <sys/mman.h>
#include <stdio.h>

int main(int argc, char **argv) {
  void *Map = mmap((void *)0x0000016000000000ULL, 0x1000, PROT_READ,
                   MAP_ANON|MAP_PRIVATE|MAP_FIXED, -1, 0);
  if (Map == (void *)-1)
    fprintf(stderr, "map failed\n");
  else
    fprintf(stderr, "mapped %p\n", Map);
  Map = mmap((void *)0x0000016000000000ULL, 0x1000, PROT_READ,
                   MAP_ANON|MAP_PRIVATE, -1, 0);
  fprintf(stderr, "mapped %p\n", Map);
  // CHECK:      in esan::initializeLibrary
  // (There can be a re-exec for stack limit here.)
  // CHECK:      Shadow scale=2 offset=0x440000000000
  // CHECK-NEXT: Shadow #0: [110000000000-114000000000) (256GB)
  // CHECK-NEXT: Shadow #1: [124000000000-12c000000000) (512GB)
  // CHECK-NEXT: Shadow #2: [148000000000-150000000000) (512GB)
  // CHECK-NEXT: mmap conflict: {{.*}}
  // CHECK-NEXT: map failed
  // CHECK-NEXT: mmap conflict: {{.*}}
  // CHECK-NEXT: mapped {{.*}}
  // CHECK-NEXT: in esan::finalizeLibrary
  return 0;
}
