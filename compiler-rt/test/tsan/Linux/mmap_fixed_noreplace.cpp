// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// REQUIRES: x86_64-target-arch

#include <assert.h>
#include <errno.h>
#include <initializer_list>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>


int main() {
  // A canonical Linux x86-64 address outside TSan's application memory.
  void *address = reinterpret_cast<void *>(0x400000000000ULL);
  const int flags = MAP_PRIVATE | MAP_ANONYMOUS;
  for (int fixed : {MAP_FIXED, MAP_FIXED_NOREPLACE}) {
    errno = 0;
    void *result = mmap(address, 4096, PROT_NONE, flags | fixed, -1, 0);
    assert(result == MAP_FAILED);
    assert(errno == EINVAL);
  }

  // An ordinary hint may still be changed to a supported application address.
  void *result = mmap(address, 4096, PROT_NONE, flags, -1, 0);
  assert(result != MAP_FAILED);
  assert(result != address);
  assert(munmap(result, 4096) == 0);
  fprintf(stderr, "DONE\n");
}

// CHECK: DONE
