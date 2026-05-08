// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void* operator new(__SIZE_TYPE__, void* p) noexcept { return p; }

void test(void *p) {
  new (p) int();
}

// CIR:     cir.func{{.*}} @_Z4testPv
// CIR-NEXT:    %[[P:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["p", init] {alignment = 8 : i64}
// CIR-NEXT:    cir.store %arg0, %[[P]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR-NEXT:    %[[P1:.*]] = cir.load align(8) %[[P]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR-NEXT:    %[[P2:.*]] = cir.cast bitcast %[[P1]] : !cir.ptr<!void> -> !cir.ptr<!s32i>
// CIR-NEXT:    %[[P3:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT:    cir.store align(4) %[[P3]], %[[P2]] : !s32i, !cir.ptr<!s32i>

// LLVM:    define{{.*}} void @_Z4testPv(ptr{{.*}} %[[ARG:.*]])
// LLVM-NEXT:   %[[P:.*]] = alloca ptr, i64 1, align 8
// LLVM-NEXT:   store ptr %[[ARG]], ptr %[[P]], align 8
// LLVM-NEXT:   %[[P1:.*]] = load ptr, ptr %[[P]], align 8
// LLVM-NEXT:   store i32 0, ptr %[[P1]], align 4
// LLVM-NEXT:   ret void

// OGCG:    define{{.*}} void @_Z4testPv(ptr{{.*}} %[[ARG:.*]])
// OGCG-NEXT:   {{.*}}:
// OGCG-NEXT:   %[[P:.*]] = alloca ptr, align 8
// OGCG-NEXT:   store ptr %[[ARG]], ptr %[[P]], align 8
// OGCG-NEXT:   %[[P1:.*]] = load ptr, ptr %[[P]], align 8
// OGCG-NEXT:   store i32 0, ptr %[[P1]], align 4
// OGCG-NEXT:   ret void

void test_complex(void *p) { new (p) int _Complex(); }

// CIR: cir.func{{.*}} @_Z12test_complexPv
// CIR:   %[[P_ADDR:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["p", init]
// CIR:   cir.store %[[ARG_0:.*]], %[[P_ADDR:.*]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:   %[[TMP_P:.*]] = cir.load {{.*}} %[[P_ADDR]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:   %[[P_COMPLEX:.*]] = cir.cast bitcast %[[TMP_P:.*]] : !cir.ptr<!void> -> !cir.ptr<!cir.complex<!s32i>>
// CIR:   %[[CONST_0:.*]] = cir.const #cir.zero : !cir.complex<!s32i>
// CIR:   cir.store {{.*}} %[[CONST_0]], %[[P_COMPLEX]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: define{{.*}} void @_Z12test_complexPv(ptr{{.*}} %[[ARG_0:.*]])
// LLVM:   %[[P_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[ARG_0]], ptr %[[P_ADDR]], align 8
// LLVM:   %[[TMP_P:.*]] = load ptr, ptr %[[P_ADDR]], align 8
// LLVM:   store { i32, i32 } zeroinitializer, ptr %[[TMP_P]], align 4

// OGCG: define{{.*}} void @_Z12test_complexPv(ptr{{.*}} %[[ARG_0:.*]])
// OGCG:   %[[P_ADDR:.*]] = alloca ptr, align 8
// OGCG:   store ptr %[[ARG_0]], ptr %[[P_ADDR]], align 8
// OGCG:   %[[TMP_P:.*]] = load ptr, ptr %[[P_ADDR]], align 8
// OGCG:   %[[P_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[TMP_P]], i32 0, i32 0
// OGCG:   %[[P_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[TMP_P]], i32 0, i32 1
// OGCG:   store i32 0, ptr %[[P_REAL_PTR]], align 4
// OGCG:   store i32 0, ptr %[[P_IMAG_PTR]], align 4
