// RUN: %clang_esan_frag -O0 %s -o %t 2>&1
// RUN: %env_esan_opts=verbosity=1 %run %t 2>&1 | FileCheck --check-prefix=%arch --check-prefix=CHECK %s

#include <unistd.h>
#include <sys/mman.h>
#include <stdio.h>

int main(int argc, char **argv) {
#if defined(__mips64)
  void *Map = mmap((void *)0x0000001600000000ULL, 0x1000, PROT_READ,
                   MAP_ANON|MAP_PRIVATE|MAP_FIXED, -1, 0);
#else
  void *Map = mmap((void *)0x0000016000000000ULL, 0x1000, PROT_READ,
                   MAP_ANON|MAP_PRIVATE|MAP_FIXED, -1, 0);
#endif
  if (Map == (void *)-1)
    fprintf(stderr, "map failed\n");
  else
    fprintf(stderr, "mapped %p\n", Map);
#if defined(__mips64)
  Map = mmap((void *)0x0000001600000000ULL, 0x1000, PROT_READ,
                   MAP_ANON|MAP_PRIVATE, -1, 0);
#else
  Map = mmap((void *)0x0000016000000000ULL, 0x1000, PROT_READ,
                   MAP_ANON|MAP_PRIVATE, -1, 0);
#endif
  fprintf(stderr, "mapped %p\n", Map);
  // CHECK:      in esan::initializeLibrary
  // (There can be a re-exec for stack limit here.)
  // x86_64:      Shadow scale=2 offset=0x440000000000
  // x86_64-NEXT: Shadow #0: [110000000000-114000000000) (256GB)
  // x86_64-NEXT: Shadow #1: [124000000000-12c000000000) (512GB)
  // x86_64-NEXT: Shadow #2: [148000000000-150000000000) (512GB)
  // mips64:      Shadow scale=2 offset=0x4400000000
  // mips64-NEXT: Shadow #0: [1140000000-1180000000) (1GB)
  // mips64-NEXT: Shadow #1: [1380000000-13c0000000) (1GB)
  // mips64-NEXT: Shadow #2: [14c0000000-1500000000) (1GB)
  // CHECK-NEXT: mmap conflict: {{.*}}
  // CHECK-NEXT: map failed
  // CHECK-NEXT: mmap conflict: {{.*}}
  // CHECK-NEXT: mapped {{.*}}
  // CHECK-NEXT: in esan::finalizeLibrary
  return 0;
}
