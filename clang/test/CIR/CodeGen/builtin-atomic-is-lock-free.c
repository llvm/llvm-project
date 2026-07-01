// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

typedef struct { char buf[24]; } Big;

int test_atomic(Big *p) { return __atomic_is_lock_free(sizeof(Big), p); }

int test_c11(void) { return __c11_atomic_is_lock_free(sizeof(Big)); }

int test_atomic_var(unsigned long n, void *p) {
  return __atomic_is_lock_free(n, p);
}

// CIR-DAG: cir.func private dso_local @__atomic_is_lock_free(!u64i, !cir.ptr<!void>) -> !cir.bool

// CIR-LABEL: cir.func{{.*}} @test_atomic(
// CIR:   %[[SZ:.*]] = cir.const #cir.int<24> : !u64i
// CIR:   %[[P:.*]] = cir.load{{.*}} : !cir.ptr<!cir.ptr<!rec_Big>>, !cir.ptr<!rec_Big>
// CIR:   %[[VP:.*]] = cir.cast bitcast %[[P]] : !cir.ptr<!rec_Big> -> !cir.ptr<!void>
// CIR:   cir.call @__atomic_is_lock_free(%[[SZ]], %[[VP]]) : (!u64i, !cir.ptr<!void>) -> !cir.bool

// CIR-LABEL: cir.func{{.*}} @test_c11(
// CIR:   %[[SZ2:.*]] = cir.const #cir.int<24> : !u64i
// CIR:   %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR:   cir.call @__atomic_is_lock_free(%[[SZ2]], %[[NULL]]) : (!u64i, !cir.ptr<!void>) -> !cir.bool

// CIR-LABEL: cir.func{{.*}} @test_atomic_var(
// CIR:   cir.call @__atomic_is_lock_free(%{{.*}}, %{{.*}}) : (!u64i, !cir.ptr<!void>) -> !cir.bool

// LLVM-LABEL: define dso_local i32 @test_atomic(
// LLVMCIR:      call i1 @__atomic_is_lock_free(i64 24, ptr %{{.*}})
// OGCG:         call zeroext i1 @__atomic_is_lock_free(i64 noundef 24, ptr noundef %{{.*}})
// LLVM:         zext i1 %{{.*}} to i32

// LLVM-LABEL: define dso_local i32 @test_c11(
// LLVMCIR:      call i1 @__atomic_is_lock_free(i64 24, ptr null)
// OGCG:         call zeroext i1 @__atomic_is_lock_free(i64 noundef 24, ptr noundef null)
// LLVM:         zext i1 %{{.*}} to i32

// LLVM-LABEL: define dso_local i32 @test_atomic_var(
// LLVMCIR:      call i1 @__atomic_is_lock_free(i64 %{{.*}}, ptr %{{.*}})
// OGCG:         call zeroext i1 @__atomic_is_lock_free(i64 noundef %{{.*}}, ptr noundef %{{.*}})
