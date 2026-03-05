// RUN: %clang_cc1 -ffreestanding  -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// This test mimics clang/test/CodeGen/X86/rd-builtins.c, which eventually
// CIR shall be able to support fully.

#include <x86intrin.h>

int test_rdtsc(void) {
  // CIR-LABEL: @test_rdtsc
  // LLVM-LABEL: @test_rdtsc
  return __rdtsc();
  // CIR: {{%.*}} = cir.llvm.intrinsic "x86.rdtsc"  : () -> !u64i
  // LLVM: call i64 @llvm.x86.rdtsc
}

unsigned long long test_rdtscp(unsigned int *a) {

  return __rdtscp(a);

  // CIR-LABEL: @__rdtscp
  // CIR: [[RDTSCP:%.*]] = cir.llvm.intrinsic "x86.rdtscp"  : () -> !rec_anon_struct
  // CIR: [[TSC_AUX:%.*]] = cir.extract_member [[RDTSCP]][1] : !rec_anon_struct -> !u32i
  // CIR: cir.store [[TSC_AUX]], %{{.*}} : !u32i, !cir.ptr<!u32i>
  // CIR: {{%.*}} = cir.extract_member [[RDTSCP]][0] : !rec_anon_struct -> !u64i

  // LLVM: @test_rdtscp
  // LLVM: [[RDTSCP:%.*]] = call { i64, i32 } @llvm.x86.rdtscp
  // LLVM: [[TSC_AUX:%.*]] = extractvalue { i64, i32 } [[RDTSCP]], 1
  // LLVM: store i32 [[TSC_AUX]], ptr %{{.*}}
  // LLVM: [[TSC:%.*]] = extractvalue { i64, i32 } [[RDTSCP]], 0
}
