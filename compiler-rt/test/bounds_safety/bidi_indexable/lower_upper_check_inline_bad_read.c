// This is a variant of `lower_upper_check` that requests inlining of `bad_read`
// that can lead to LLVM discovering UB in soft trap mode.
// RUN: %clang_bsafe %s -o %t
// RUN: %expect-no-trap %t
// RUN: %expect-trap --verify-prefix=lower-trap %s %t arg1
// RUN: %expect-trap --verify-prefix=upper-trap %s %t arg1 arg2
#include <ptrcheck.h>
#include <stdio.h>
#include "soft_trap_runtime_impl.h"

// lower-trap-merged{bad_read}
// upper-trap-merged{bad_read}
__attribute__((always_inline)) int bad_read(int *__bidi_indexable ptr, int idx) {
  // lower-trap@+2{indexing below lower bound in 'ptr[idx]'}
  // upper-trap@+1{indexing above upper bound in 'ptr[idx]'}
  return ptr[idx];
}

int main(int argc, const char **__counted_by(argc) argv) {
  int pad;
  int local[] = {0, 1};
  int pad2;
  int result = 0;
  if (argc == 1) {
    result = bad_read(local, 1);
  } else if (argc == 2) {
    result = bad_read(local, -1);
  } else {
    result = bad_read(local, 2);
  }
  printf("result: %d\n", result);
  return 0;
}
