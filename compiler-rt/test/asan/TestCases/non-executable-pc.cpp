// RUN: %clangxx_asan %s -o %t
// RUN: not %run %t 0 2>&1 | FileCheck %s
// RUN: not %run %t n 2>&1 | FileCheck %s -check-prefix=CHECK -check-prefix=NON_EXEC

// Not every OS lists every memory region in MemoryMappingLayout.
// This is limited to x86_64 because some architectures (e.g. the s390 before
// the z14) don't support NX mappings and others like PowerPC use function
// descriptors.
// REQUIRES: x86-target-arch && (linux || freebsd || netbsd)

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
  // CHECK: DEADLYSIGNAL
  // CHECK: {{AddressSanitizer: (SEGV|access-violation).*(address|pc) }}
  // NON_EXEC: PC is at a non-executable region. Maybe a wild jump?
  return 0;
}
