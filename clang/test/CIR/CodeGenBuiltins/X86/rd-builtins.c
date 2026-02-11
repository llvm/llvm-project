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
