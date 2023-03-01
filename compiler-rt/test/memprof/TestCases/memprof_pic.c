// REQUIRES: x86_64-linux
// RUN: %clang_memprof -fuse-ld=lld -Wl,--no-rosegment -gmlt -fdebug-info-for-profiling -fmemory-profile -mno-omit-leaf-frame-pointer -fno-omit-frame-pointer -fno-optimize-sibling-calls -m64 -Wl,-build-id -pie %s -o %t.memprofexe
// RUN: env MEMPROF_OPTIONS=log_path=stdout %t.memprofexe > %t.memprofraw
// RUN: not %llvm_profdata show --memory %t.memprofraw --profiled-binary %t.memprofexe -o - 2>&1 | FileCheck %s
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  char *x = (char *)malloc(10);
  memset(x, 0, 10);
  free(x);
  x = (char *)malloc(10);
  memset(x, 0, 10);
  free(x);
  return 0;
}

// COM: This test ensures that llvm-profdata fails with a descriptive error message
// COM: when invoked on a memprof profiled binary which was built with position
// COM: independent code.
// CHECK: Unsupported position independent code
