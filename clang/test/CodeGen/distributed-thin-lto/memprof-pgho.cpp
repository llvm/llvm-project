// Test end-to-end ThinLTO optimization pipeline with PGHO, that it does not
// interfere with other allocation instrumentation features.
//
// REQUIRES: x86-registered-target
//
// RUN: split-file %s %t
// RUN: llvm-profdata merge %t/memprof.yaml -o %t/use.memprofdata
//
// RUN: %clangxx --target=x86_64-linux-gnu -O2 -flto=thin -g -fmemory-profile-use=%t/use.memprofdata %t/src.cpp -c -o %t.o
// RUN: llvm-lto2 run %t.o -thinlto-distributed-indexes -supports-hot-cold-new -r=%t.o,main,plx -r=%t.o,_Z3foov,plx -r=%t.o,_Znam, -o %t.out
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O1 -x ir %t.o -fthinlto-index=%t.o.thinlto.bc -mllvm -optimize-hot-cold-new -emit-llvm -o - 2>&1 | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O2 -x ir %t.o -fthinlto-index=%t.o.thinlto.bc -mllvm -optimize-hot-cold-new -emit-llvm -o - 2>&1 | FileCheck %s --check-prefixes=CHECK,DEFAULT
//
// RUN: %clangxx --target=x86_64-linux-gnu -O2 -flto=thin -g -fsanitize=alloc-token -falloc-token-max=32 -fmemory-profile-use=%t/use.memprofdata %t/src.cpp -c -o %t.o
// RUN: llvm-lto2 run %t.o -thinlto-distributed-indexes -supports-hot-cold-new -r=%t.o,main,plx -r=%t.o,_Z3foov,plx -r=%t.o,_Znam, -o %t.out
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O1 -x ir %t.o -fsanitize=alloc-token -fthinlto-index=%t.o.thinlto.bc -mllvm -optimize-hot-cold-new -emit-llvm -o - 2>&1 | FileCheck %s --check-prefixes=CHECK,ALLOCTOKEN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O2 -x ir %t.o -fsanitize=alloc-token -fthinlto-index=%t.o.thinlto.bc -mllvm -optimize-hot-cold-new -emit-llvm -o - 2>&1 | FileCheck %s --check-prefixes=CHECK,ALLOCTOKEN

//--- memprof.yaml
---
HeapProfileRecords:
  - GUID: 0x7f8d88fcc70a347b
    AllocSites:
    - Callstack:
      - { Function: 0x7f8d88fcc70a347b, LineOffset: 1, Column: 10, IsInlineFrame: false }
      - { Function: 0xdb956436e78dd5fa, LineOffset: 1, Column: 13, IsInlineFrame: false }
      MemInfoBlock:
        AllocCount: 1
        TotalAccessCount: 0
        MinAccessCount: 0
        MaxAccessCount: 0
        TotalSize: 10
        MinSize: 10
        MaxSize: 10
        AllocTimestamp: 100
        DeallocTimestamp: 100
        TotalLifetime: 100000
        MinLifetime: 100000
        MaxLifetime: 100000
        AllocCpuId: 0
        DeallocCpuId: 0
        NumMigratedCpu: 0
        NumLifetimeOverlaps: 0
        NumSameAllocCpu: 0
        NumSameDeallocCpu: 0
        DataTypeId: 0
        TotalAccessDensity: 0
        MinAccessDensity: 0
        MaxAccessDensity: 0
        TotalLifetimeAccessDensity: 0
        MinLifetimeAccessDensity: 0
        MaxLifetimeAccessDensity: 0
        AccessHistogramSize: 0
        AccessHistogram: 0
...

//--- src.cpp
// CHECK-LABEL: define{{.*}} ptr @_Z3foov()
// DEFAULT:    call {{.*}} ptr @_Znam12__hot_cold_t(i64 10, i8 -128)
// ALLOCTOKEN: call {{.*}} ptr @__alloc_token__Znam12__hot_cold_t(i64 10, i8 -128, i64 12){{.*}} !alloc_token
char *foo() {
  return new char[10];
}

int main() {
  char *a = foo();
  delete[] a;
  return 0;
}
