// REQUIRES: x86_64-linux
// RUN: %clang_memprof -fuse-ld=lld -Wl,--no-rosegment -gmlt -fdebug-info-for-profiling -fmemory-profile -mno-omit-leaf-frame-pointer -fno-omit-frame-pointer -fno-optimize-sibling-calls -m64 -Wl,-build-id -no-pie %s -o %t.memprofexe
// RUN: env MEMPROF_OPTIONS=log_path=stdout %t.memprofexe > %t.memprofraw
// RUN: %llvm_profdata show --memory %t.memprofraw --profiled-binary %t.memprofexe -o - | FileCheck %s
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

// COM: We expect 2 MIB entries, 1 each for the malloc calls in the program. Any
// COM: additional allocations which do not originate from the main binary are pruned.

// CHECK:  MemprofProfile:
// CHECK-NEXT:   Summary:
// CHECK-NEXT:     Version: 2
// CHECK-NEXT:     NumSegments: {{[0-9]+}}
// CHECK-NEXT:     NumMibInfo: 2
// CHECK-NEXT:     NumAllocFunctions: 1
// CHECK-NEXT:     NumStackOffsets: 2
// CHECK-NEXT:   Segments:
// CHECK-NEXT:   -
// CHECK-NEXT:     BuildId: <None>
// CHECK-NEXT:     Start: 0x{{[0-9]+}}
// CHECK-NEXT:     End: 0x{{[0-9]+}}
// CHECK-NEXT:     Offset: 0x{{[0-9]+}}
// CHECK-NEXT:   -

// CHECK:   Records:
// CHECK-NEXT:   -
// CHECK-NEXT:     FunctionGUID: {{[0-9]+}}
// CHECK-NEXT:     AllocSites:
// CHECK-NEXT:     -
// CHECK-NEXT:       Callstack:
// CHECK-NEXT:       -
// CHECK-NEXT:         Function: {{[0-9]+}}
// CHECK-NEXT:         SymbolName: main
// CHECK-NEXT:         LineOffset: 1
// CHECK-NEXT:         Column: 21
// CHECK-NEXT:         Inline: 0
// CHECK-NEXT:       MemInfoBlock:
// CHECK-NEXT:         AllocCount: 1
// CHECK-NEXT:         TotalAccessCount: 2
// CHECK-NEXT:         MinAccessCount: 2
// CHECK-NEXT:         MaxAccessCount: 2
// CHECK-NEXT:         TotalSize: 10
// CHECK-NEXT:         MinSize: 10
// CHECK-NEXT:         MaxSize: 10
// CHECK-NEXT:         AllocTimestamp: {{[0-9]+}}
// CHECK-NEXT:         DeallocTimestamp: {{[0-9]+}}
// CHECK-NEXT:         TotalLifetime: 0
// CHECK-NEXT:         MinLifetime: 0
// CHECK-NEXT:         MaxLifetime: 0
// CHECK-NEXT:         AllocCpuId: {{[0-9]+}}
// CHECK-NEXT:         DeallocCpuId: {{[0-9]+}}
// CHECK-NEXT:         NumMigratedCpu: 0
// CHECK-NEXT:         NumLifetimeOverlaps: 0
// CHECK-NEXT:         NumSameAllocCpu: 0
// CHECK-NEXT:         NumSameDeallocCpu: 0
// CHECK-NEXT:         DataTypeId: {{[0-9]+}}
// CHECK-NEXT:         TotalAccessDensity: 20
// CHECK-NEXT:         MinAccessDensity: 20
// CHECK-NEXT:         MaxAccessDensity: 20
// CHECK-NEXT:         TotalLifetimeAccessDensity: 20000
// CHECK-NEXT:         MinLifetimeAccessDensity: 20000
// CHECK-NEXT:         MaxLifetimeAccessDensity: 20000
// CHECK-NEXT:     -
// CHECK-NEXT:       Callstack:
// CHECK-NEXT:       -
// CHECK-NEXT:         Function: {{[0-9]+}}
// CHECK-NEXT:         SymbolName: main
// CHECK-NEXT:         LineOffset: 4
// CHECK-NEXT:         Column: 15
// CHECK-NEXT:         Inline: 0
// CHECK-NEXT:       MemInfoBlock:
// CHECK-NEXT:         AllocCount: 1
// CHECK-NEXT:         TotalAccessCount: 2
// CHECK-NEXT:         MinAccessCount: 2
// CHECK-NEXT:         MaxAccessCount: 2
// CHECK-NEXT:         TotalSize: 10
// CHECK-NEXT:         MinSize: 10
// CHECK-NEXT:         MaxSize: 10
// CHECK-NEXT:         AllocTimestamp: {{[0-9]+}}
// CHECK-NEXT:         DeallocTimestamp: {{[0-9]+}}
// CHECK-NEXT:         TotalLifetime: 0
// CHECK-NEXT:         MinLifetime: 0
// CHECK-NEXT:         MaxLifetime: 0
// CHECK-NEXT:         AllocCpuId: {{[0-9]+}}
// CHECK-NEXT:         DeallocCpuId: {{[0-9]+}}
// CHECK-NEXT:         NumMigratedCpu: 0
// CHECK-NEXT:         NumLifetimeOverlaps: 0
// CHECK-NEXT:         NumSameAllocCpu: 0
// CHECK-NEXT:         NumSameDeallocCpu: 0
// CHECK-NEXT:         DataTypeId: {{[0-9]+}}
// CHECK-NEXT:         TotalAccessDensity: 20
// CHECK-NEXT:         MinAccessDensity: 20
// CHECK-NEXT:         MaxAccessDensity: 20
// CHECK-NEXT:         TotalLifetimeAccessDensity: 20000
// CHECK-NEXT:         MinLifetimeAccessDensity: 20000
// CHECK-NEXT:         MaxLifetimeAccessDensity: 20000
