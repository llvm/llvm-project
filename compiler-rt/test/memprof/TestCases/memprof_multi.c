// REQUIRES: x86_64-linux
// RUN: %clang_memprof -fuse-ld=lld -Wl,--no-rosegment -gmlt -fdebug-info-for-profiling -fmemory-profile -mno-omit-leaf-frame-pointer -fno-omit-frame-pointer -fno-optimize-sibling-calls -m64 -Wl,-build-id -no-pie %s -o %t.memprofexe
// RUN: env MEMPROF_OPTIONS=log_path=stdout %t.memprofexe > %t.memprofraw
// RUN: %llvm_profdata show --memory %t.memprofraw --profiled-binary %t.memprofexe -o - | FileCheck %s
#include <sanitizer/memprof_interface.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  char *x = (char *)malloc(10);
  memset(x, 0, 10);
  free(x);
  __memprof_profile_dump();
  x = (char *)malloc(10);
  memset(x, 0, 10);
  free(x);
  return 0;
}

// COM: We expect 2 MIB entries, 1 each for the malloc calls in the program.

// CHECK:  MemprofProfile:
// CHECK-NEXT:  Summary:
// CHECK-NEXT:    Version: 2
// CHECK-NEXT:    NumSegments: {{[0-9]+}}
// CHECK-NEXT:    NumMibInfo: 2
// CHECK-NEXT:    NumAllocFunctions: 1
// CHECK-NEXT:    NumStackOffsets: 2

// CHECK:        SymbolName: main
// CHECK-NEXT:     LineOffset: 1
// CHECK-NEXT:     Column: 21

// CHECK:        SymbolName: main
// CHECK-NEXT:     LineOffset: 5
// CHECK-NEXT:     Column: 15
