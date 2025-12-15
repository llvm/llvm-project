// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s -check-prefix=OGCG

void foo(void *a) {
  __builtin_prefetch(a);        // rw=0, locality=3
  __builtin_prefetch(a, 0);     // rw=0, locality=3
  __builtin_prefetch(a, 1);     // rw=1, locality=3
  __builtin_prefetch(a, 1, 1);  // rw=1, locality=1
}

// CIR-LABEL: cir.func dso_local @foo(
// CIR: %[[ALLOCA:.*]] = cir.alloca !cir.ptr<!void>
// CIR: cir.store %arg0, %[[ALLOCA]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR: %[[P1:.*]] = cir.load{{.*}} %[[ALLOCA]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR: cir.prefetch read locality(3) %[[P1]] : !cir.ptr<!void>
// CIR: %[[P2:.*]] = cir.load{{.*}} %[[ALLOCA]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR: cir.prefetch read locality(3) %[[P2]] : !cir.ptr<!void>
// CIR: %[[P3:.*]] = cir.load{{.*}} %[[ALLOCA]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR: cir.prefetch write locality(3) %[[P3]] : !cir.ptr<!void>
// CIR: %[[P4:.*]] = cir.load{{.*}} %[[ALLOCA]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR: cir.prefetch write locality(1) %[[P4]] : !cir.ptr<!void>
// CIR: cir.return

// LLVM-LABEL: define dso_local void @foo(
// LLVM: [[ALLOCA:%.*]] = alloca ptr, i64 1
// LLVM: store ptr {{.*}}, ptr [[ALLOCA]]
// LLVM: [[LP1:%.*]] = load ptr, ptr [[ALLOCA]]
// LLVM: call void @llvm.prefetch.p0(ptr [[LP1]], i32 0, i32 3, i32 1)
// LLVM: [[LP2:%.*]] = load ptr, ptr [[ALLOCA]]
// LLVM: call void @llvm.prefetch.p0(ptr [[LP2]], i32 0, i32 3, i32 1)
// LLVM: [[LP3:%.*]] = load ptr, ptr [[ALLOCA]]
// LLVM: call void @llvm.prefetch.p0(ptr [[LP3]], i32 1, i32 3, i32 1)
// LLVM: [[LP4:%.*]] = load ptr, ptr [[ALLOCA]]
// LLVM: call void @llvm.prefetch.p0(ptr [[LP4]], i32 1, i32 1, i32 1)
// LLVM: ret void

// OGCG-LABEL: define dso_local void @foo(ptr
// OGCG: call void @llvm.prefetch.p0(ptr {{.*}}, i32 0, i32 3, i32 1)
// OGCG: call void @llvm.prefetch.p0(ptr {{.*}}, i32 0, i32 3, i32 1)
// OGCG: call void @llvm.prefetch.p0(ptr {{.*}}, i32 1, i32 3, i32 1)
// OGCG: call void @llvm.prefetch.p0(ptr {{.*}}, i32 1, i32 1, i32 1)
// OGCG: ret void
