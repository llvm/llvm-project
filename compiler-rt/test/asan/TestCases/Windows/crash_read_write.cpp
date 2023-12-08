// RUN: %clangxx_asan -std=c++11 -O0 %s -o %t
// RUN: not %run %t       2>&1 | FileCheck %s --check-prefix=READ
// RUN: not %run %t write 2>&1 | FileCheck %s --check-prefix=WRITE

#include <windows.h>
#include <stdio.h>

static volatile int sink;
__attribute__((noinline)) void Read(int *ptr) { sink = *ptr; }
__attribute__((noinline)) void Write(int *ptr) { *ptr = 0; }
int main(int argc, char **argv) {
  // Writes to shadow are detected as reads from shadow gap (because of how the
  // shadow mapping works). This is kinda hard to fix. Test a random address in
  // the application part of the address space.
  void *volatile p = VirtualAlloc(0, 4096, MEM_COMMIT, PAGE_READONLY);
  bool ok = VirtualFree(p, 0, MEM_RELEASE);
  if (!ok) {
    printf("VirtualFree failed\n");
    return 0;
  }
  if (argc == 1)
    Read((int *)p);
  else
    Write((int *)p);
}
// READ: AddressSanitizer: access-violation on unknown address
// READ: The signal is caused by a READ memory access.
// WRITE: AddressSanitizer: access-violation on unknown address
// WRITE: The signal is caused by a WRITE memory access.
