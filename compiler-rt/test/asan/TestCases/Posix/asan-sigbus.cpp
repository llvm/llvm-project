// Check handle_bus flag
// Defaults to true
// RUN: %clangxx_asan -std=c++11 %s -o %t
// RUN: not %run %t %T/file 2>&1 | FileCheck %s -check-prefix=CHECK-BUS
// RUN: %env_asan_opts=handle_sigbus=0 not --crash %run %t %T/file 2>&1 | FileCheck %s

// UNSUPPORTED: ios

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

char array[4096];
int main(int argc, char **argv) {
  assert(argc > 1);
  int fd = open(argv[1], O_RDWR | O_CREAT, 0700);
  if (fd < 0) {
    perror("open");
    exit(1);
  }
  assert(write(fd, array, sizeof(array)) == sizeof(array));

  // Write some zeroes to the file, then mmap it while it has a 4KiB size
  char *addr = (char *)mmap(nullptr, sizeof(array), PROT_READ,
                            MAP_FILE | MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED) {
    perror("mmap");
    exit(1);
  }

  // Truncate the file so our memory isn't valid any more
  assert(ftruncate(fd, 0) == 0);

  // Try to access the memory
  return addr[42];
  // CHECK-NOT: DEADLYSIGNAL
  // CHECK-BUS: DEADLYSIGNAL
  // CHECK-BUS: ERROR: AddressSanitizer: BUS
}
