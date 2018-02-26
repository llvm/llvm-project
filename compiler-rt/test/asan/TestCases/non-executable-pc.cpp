// RUN: %clangxx_asan %s -o %t
// RUN: not %run %t 0 2>&1 | FileCheck %s
// RUN: not %run %t n 2>&1 | FileCheck %s -check-prefix=CHECK -check-prefix=NON_EXEC

// Not every OS lists every memory region in MemoryMappingLayout.
// REQUIRES: linux || freebsd || netbsd

#include <assert.h>

typedef void void_f();
int main(int argc, char **argv) {
  char *array = new char[42];
  void_f *func;
  assert(argc > 1);
  if (argv[1][0] == '0') {
    func = (void_f *)0x04;
  } else {
    assert(argv[1][0] == 'n');
    func = (void_f *)array;
  }

  func();
  // x86 reports the SEGV with both address=X and pc=X.
  // On PowerPC64 ELFv1, the pointer is taken to be a function-descriptor
  // pointer out of which three 64-bit quantities are read. This will SEGV, but
  // the compiler is free to choose the order. As a result, the address is
  // either X, X+0x8 or X+0x10. The pc is still in main() because it has not
  // actually made the call when the faulting access occurs.
  // CHECK: DEADLYSIGNAL
  // CHECK: {{AddressSanitizer: (SEGV|access-violation).*(address|pc) }}
  // NON_EXEC: PC is at a non-executable region. Maybe a wild jump?
  return 0;
}
