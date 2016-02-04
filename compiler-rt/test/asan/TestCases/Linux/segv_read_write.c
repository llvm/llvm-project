// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run %t       2>&1 | FileCheck %s --check-prefix=READ
// RUN: not %run %t write 2>&1 | FileCheck %s --check-prefix=WRITE

static volatile int sink;
__attribute__((noinline)) void Read(int *ptr) { sink = *ptr; }
__attribute__((noinline)) void Write(int *ptr) { *ptr = 0; }
int main(int argc, char **argv) {
  if (argc == 1)
    Read((int *)0);
  else
    Write((int *)0);
}
// READ: AddressSanitizer: SEGV on unknown address
// READ: The signal is caused by a READ memory access.
// WRITE: AddressSanitizer: SEGV on unknown address
// WRITE: The signal is caused by a WRITE memory access.
