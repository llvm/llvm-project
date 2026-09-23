// Test end-to-end optimization pipeline with PGHO, that it does not interfere
// with other allocation instrumentation features.
//
// RUN: split-file %s %t
// RUN: llvm-profdata merge %t/memprof.yaml -o %t/use.profdata
// RUN: %clang_cc1 -O2 -debug-info-kind=limited -fmemory-profile-use=%t/use.profdata -mllvm -optimize-hot-cold-new \
// RUN:            %t/src.cpp -triple x86_64-linux-gnu -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: %clang_cc1 -O2 -fsanitize=alloc-token -debug-info-kind=limited -fmemory-profile-use=%t/use.profdata -mllvm -optimize-hot-cold-new \
// RUN:             %t/src.cpp -triple x86_64-linux-gnu -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,ALLOCTOKEN

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
// ALLOCTOKEN: call {{.*}} ptr @__alloc_token__Znam12__hot_cold_t(i64 10, i8 -128, i64 1538840549748785101){{.*}} !alloc_token
char *foo() {
  return new char[10];
}

int main() {
  char *a = foo();
  delete[] a;
  return 0;
}
