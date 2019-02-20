// RUN: %clang_hwasan %s -DTEST_NO=1 -mllvm -hwasan-instrument-mem-intrinsics -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=WRITE
// RUN: %clang_hwasan %s -DTEST_NO=2 -mllvm -hwasan-instrument-mem-intrinsics -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=READ
// RUN: %clang_hwasan %s -DTEST_NO=3 -mllvm -hwasan-instrument-mem-intrinsics -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=WRITE
// RUN: %clang_hwasan %s -DTEST_NO=2 -mllvm -hwasan-instrument-mem-intrinsics -o %t && not %env_hwasan_opts=halt_on_error=0 %run %t 2>&1 | FileCheck %s --check-prefix=RECOVER

// REQUIRES: stable-runtime

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
  char Q[16];
  char P[16];
#if TEST_NO == 1
  memset(Q, 0, 32);
#elif TEST_NO == 2
  memmove(Q, Q + 16, 16);
#elif TEST_NO == 3
  memcpy(Q, P, 32);
#endif
  write(STDOUT_FILENO, "recovered\n", 10);
  // WRITE: ERROR: HWAddressSanitizer: tag-mismatch on address
  // WRITE: WRITE {{.*}} tags: [[PTR_TAG:..]]/[[MEM_TAG:..]] (ptr/mem)
  // WRITE: Memory tags around the buggy address (one tag corresponds to 16 bytes):
  // WRITE: =>{{.*}}[[MEM_TAG]]
  // WRITE-NOT: recovered

  // READ: ERROR: HWAddressSanitizer: tag-mismatch on address
  // READ: READ {{.*}} tags: [[PTR_TAG:..]]/[[MEM_TAG:..]] (ptr/mem)
  // READ: Memory tags around the buggy address (one tag corresponds to 16 bytes):
  // READ: =>{{.*}}[[MEM_TAG]]
  // READ-NOT: recovered

  // RECOVER: recovered
  return 0;
}
