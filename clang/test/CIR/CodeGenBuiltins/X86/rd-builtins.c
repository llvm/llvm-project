// RUN: %clang -target x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang -target x86_64-unknown-linux-gnu -fclangir -S -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang -target x86_64-unknown-linux-gnu -S -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

#include <x86intrin.h>

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
