// RUN: %clang %s -o %t && %run %t

#include <assert.h>
#include <sys/mman.h>

int main() {
  char *buf = (char *)mmap(0, 100000, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(buf);
  munmap(buf, 100000);
}
