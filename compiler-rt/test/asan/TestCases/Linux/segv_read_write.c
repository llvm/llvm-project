// RUN: %clangxx_asan -std=c++11 -O0 %s -o %t
// RUN: not %run %t       2>&1 | FileCheck %s --check-prefix=READ
// RUN: not %run %t write 2>&1 | FileCheck %s --check-prefix=WRITE
// UNSUPPORTED: powerpc64,mips

#include <sys/mman.h>

static volatile int sink;
__attribute__((noinline)) void Read(int *ptr) { sink = *ptr; }
__attribute__((noinline)) void Write(int *ptr) { *ptr = 0; }
int main(int argc, char **argv) {
  // Writes to shadow are detected as reads from shadow gap (because of how the
  // shadow mapping works). This is kinda hard to fix. Test a random address in
  // the application part of the address space.
  void *volatile p =
      mmap(nullptr, 4096, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
  munmap(p, 4096);
  if (argc == 1)
    Read((int *)p);
  else
    Write((int *)p);
}
// READ: AddressSanitizer: SEGV on unknown address
// READ: The signal is caused by a READ memory access.
// WRITE: AddressSanitizer: SEGV on unknown address
// WRITE: The signal is caused by a WRITE memory access.
