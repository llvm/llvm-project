// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

#include <x86intrin.h>

// CIR-LABEL: @__rdpmc
// CIR: cir.call_llvm_intrinsic "x86.rdpmc" %{{.*}} : (!s32i) -> !u64i

unsigned long long test_rdpmc(int a) {
    // CIR-LABEL: test_rdpmc
    // CIR: cir.call @__rdpmc
    // CIR: cir.store %{{.*}}, %{{.*}} : !u64i, !cir.ptr<!u64i>
    // CIR: cir.return %{{.*}} : !u64i

    // LLVM-LABEL: @test_rdpmc
    // LLVM: call i64 @llvm.x86.rdpmc
    // LLVM: store i64 %{{.*}}, ptr %{{.*}}, align 8
    // LLVM: ret i64 %{{.*}}

    // OGCG-LABEL: @test_rdpmc
    // OGCG: call i64 @llvm.x86.rdpmc
    // OGCG: ret i64 %{{.*}}
    return _rdpmc(a);
}

unsigned long long test_rdtsc(void) {
  // CIR-LABEL: @test_rdtsc
  // CIR: %{{.*}} = cir.call_llvm_intrinsic "x86.rdtsc" : () -> !u64i

  // LLVM-LABEL: @test_rdtsc
  // LLVM: call i64 @llvm.x86.rdtsc()

  // OGCG-LABEL: @test_rdtsc
  // OGCG: call i64 @llvm.x86.rdtsc()

  return __rdtsc();
}

unsigned long long test_rdtscp(unsigned int *a) {
  // CIR-LABEL: @test_rdtscp
  // CIR: %[[RDTSCP:.*]] = cir.call_llvm_intrinsic "x86.rdtscp" : () -> !rec_anon_struct
  // CIR: %[[TSC_AUX:.*]] = cir.extract_member %[[RDTSCP]][1] : !rec_anon_struct -> !u32i
  // CIR: cir.store {{.*}}%[[TSC_AUX]], {{%.*}} : !u32i
  // CIR: %[[TSC:.*]] = cir.extract_member %[[RDTSCP]][0] : !rec_anon_struct -> !u64i

  // LLVM-LABEL: @test_rdtscp
  // LLVM: %[[RDTSCP:.*]] = call { i64, i32 } @llvm.x86.rdtscp()
  // LLVM: %[[TSC_AUX:.*]] = extractvalue { i64, i32 } %[[RDTSCP]], 1
  // LLVM: store i32 %[[TSC_AUX]], ptr %{{.*}}
  // LLVM: %[[TSC:.*]] = extractvalue { i64, i32 } %[[RDTSCP]], 0

  // OGCG-LABEL: @test_rdtscp
  // OGCG: %[[RDTSCP:.*]] = call { i64, i32 } @llvm.x86.rdtscp
  // OGCG: %[[TSC_AUX:.*]] = extractvalue { i64, i32 } %[[RDTSCP]], 1
  // OGCG: store i32 %[[TSC_AUX]], ptr %{{.*}}
  // OGCG: %[[TSC:.*]] = extractvalue { i64, i32 } %[[RDTSCP]], 0

  return __builtin_ia32_rdtscp(a);
}

