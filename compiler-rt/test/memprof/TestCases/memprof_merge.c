// REQUIRES: x86_64-linux
// COM: Name compression disabled since some buildbots do not have zlib.
// RUN: %clang -mllvm -enable-name-compression=false -fprofile-generate %s -o %t.instr.out
// RUN: env LLVM_PROFILE_FILE=%t.profraw %t.instr.out
// RUN: %clang_memprof -fuse-ld=lld -Wl,--no-rosegment -gmlt -fdebug-info-for-profiling -fmemory-profile -mno-omit-leaf-frame-pointer -fno-omit-frame-pointer -fno-optimize-sibling-calls -m64 -Wl,-build-id -no-pie %s -o %t.memprofexe
// RUN: env MEMPROF_OPTIONS=log_path=stdout %t.memprofexe > %t.memprofraw
// RUN: %llvm_profdata merge %t.profraw %t.memprofraw --profiled-binary %t.memprofexe -o %t.prof
// RUN: %llvm_profdata show %t.prof | FileCheck %s
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

// COM: For now we only check the validity of the instrumented profile since we don't
// COM: have a way to display the contents of the memprof indexed format yet.

// CHECK: Instrumentation level: IR  entry_first = 0
// CHECK: Total functions: 1
// CHECK: Maximum function count: 1
// CHECK: Maximum internal block count: 0
