// RUN: %clang -target x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang -target x86_64-unknown-linux-gnu -fclangir -S -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang -target x86_64-unknown-linux-gnu -S -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

#include <x86intrin.h>

unsigned long long test_rdpmc(int a) {
// CIR-LABEL: test_rdpmc
// CIR: %{{.*}} = cir.call @__rdpmc(%{{.*}}) : (!s32i) -> !u64i
// CIR: cir.store %{{.*}} : !u64i, !cir.ptr<!u64i>
// CIR: cir.return %{{.*}} : !u64i

// LLVM-LABEL: test_rdpmc
// LLVM: %{{.*}} = call i64 @llvm.x86.rdpmc(i32 %{{.*}})
// LLVM: store i64 %{{.*}}, ptr %{{.*}}
// LLVM: ret i64 %{{.*}}

// OGCG-LABEL: test_rdpmc
// OGCG: %{{.*}} = call i64 @llvm.x86.rdpmc(i32 %{{.*}})
// OGCG: ret i64 %{{.*}}
  return _rdpmc(a);
}

int test_rdtsc(void) {
// CIR-LABEL: test_rdtsc
// CIR: cir.call_llvm_intrinsic "x86.rdtsc"
// CIR: cir.cast integral %{{.*}} : !u64i -> !s32i
// CIR: cir.return %{{.*}} : !u64i

// LLVM-LABEL: test_rdtsc
// LLVM: %{{.*}} = call i64 @llvm.x86.rdtsc()
// LLVM: %{{.*}} = trunc i64 %{{.*}} to i32
// LLVM: ret i32 %{{.*}}

// OGCG-LABEL: test_rdtsc
// OGCG: %{{.*}} = call i64 @llvm.x86.rdtsc()
// OGCG: %{{.*}} = trunc i64 %{{.*}} to i32
// OGCG: ret i32 %{{.*}}

  return _rdtsc();
}

unsigned long long test_rdtscp(unsigned int *a) {
// CIR-LABEL: test_rdtscp
// CIR: %{{.*}} = cir.call @__rdtscp(%{{.*}}) : (!cir.ptr<!u32i>) -> !u64i
// CIR: cir.store %{{.*}} : !u64i, !cir.ptr<!u64i>
// CIR: cir.return %{{.*}} : !u64i

// LLVM-LABEL: test_rdtscp
// LLVM: %{{.*}} = call { i64, i32 } @llvm.x86.rdtscp()
// LLVM: %{{.*}} = extractvalue { i64, i32 } %{{.*}}, 1
// LLVM: store i32 %{{.*}}, ptr %{{.*}}
// LLVM: %{{.*}} = extractvalue { i64, i32 } %{{.*}}, 0
// LLVM: ret i64 %{{.*}}

// OGCG-LABEL: test_rdtscp
// OGCG: %{{.*}} = call { i64, i32 } @llvm.x86.rdtscp()
// OGCG: %{{.*}} = extractvalue { i64, i32 } %{{.*}}, 1
// OGCG: store i32 %{{.*}}, ptr %{{.*}}
// OGCG: %{{.*}} = extractvalue { i64, i32 } %{{.*}}, 0
// OGCG: ret i64 %{{.*}}
  return __rdtscp(a);
}
