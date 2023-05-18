// RUN: %clang %s -o %t && %run %t

#define _LARGEFILE64_SOURCE 1

#include <assert.h>
#include <sys/mman.h>

int main() {
  char *buf = (char *)mmap64(0, 100000, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(buf);
  munmap(buf, 100000);
}
