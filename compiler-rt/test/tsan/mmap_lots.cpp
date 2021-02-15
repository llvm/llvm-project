// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

// Test that mmap does not return unexpected addresses
// (the check is in the interceptor).

#include <fcntl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

int main() {
  int fd = open("/dev/zero", O_RDWR);
  if (fd == -1) perror("open(/dev/zero)"), exit(1);
  for (size_t mmap_size = 64ull << 30; mmap_size >= 4 << 10; mmap_size /= 2) {
    size_t allocated = 0;
    while (mmap(0, mmap_size, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE,
                fd, 0) != MAP_FAILED) {
      allocated += mmap_size;
    }
    fprintf(stderr, "allocated %zu with size %zu\n", allocated, mmap_size);
  }
  fprintf(stderr, "DONE\n");
}

// CHECK: DONE
