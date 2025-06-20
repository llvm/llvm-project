// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu %s -fclangir -emit-cir -o %t.cir
// RUN: FileCheck --input-file=%t.cir -check-prefix=CIR %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu %s -fclangir -emit-llvm -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll -check-prefix=LLVM %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu %s -emit-llvm -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=OGCG %s

void t1() {
  int *p1 = nullptr;
  int *p2 = 0;
  int *p3 = (int*)0;
}

// CIR:      cir.func @_Z2t1v()
// CIR-NEXT:     %[[P1:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["p1", init] {alignment = 8 : i64}
// CIR-NEXT:     %[[P2:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["p2", init] {alignment = 8 : i64}
// CIR-NEXT:     %[[P3:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["p3", init] {alignment = 8 : i64}
// CIR-NEXT:     %[[NULLPTR1:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
// CIR-NEXT:     cir.store{{.*}} %[[NULLPTR1]], %[[P1]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR-NEXT:     %[[NULLPTR2:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
// CIR-NEXT:     cir.store{{.*}} %[[NULLPTR2]], %[[P2]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR-NEXT:     %[[NULLPTR3:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
// CIR-NEXT:     cir.store{{.*}} %[[NULLPTR3]], %[[P3]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR-NEXT:     cir.return
// CIR-NEXT: }

// LLVM:      define{{.*}} @_Z2t1v()
// LLVM-NEXT:     %[[P1:.*]] = alloca ptr, i64 1, align 8
// LLVM-NEXT:     %[[P2:.*]] = alloca ptr, i64 1, align 8
// LLVM-NEXT:     %[[P3:.*]] = alloca ptr, i64 1, align 8
// LLVM-NEXT:     store ptr null, ptr %[[P1]], align 8
// LLVM-NEXT:     store ptr null, ptr %[[P2]], align 8
// LLVM-NEXT:     store ptr null, ptr %[[P3]], align 8
// LLVM-NEXT:     ret void
// LLVM-NEXT: }

// OGCG:      define{{.*}} @_Z2t1v()
// OGCG-NEXT: entry:
// OGCG-NEXT:     %[[P1:.*]] = alloca ptr, align 8
// OGCG-NEXT:     %[[P2:.*]] = alloca ptr, align 8
// OGCG-NEXT:     %[[P3:.*]] = alloca ptr, align 8
// OGCG-NEXT:     store ptr null, ptr %[[P1]], align 8
// OGCG-NEXT:     store ptr null, ptr %[[P2]], align 8
// OGCG-NEXT:     store ptr null, ptr %[[P3]], align 8
// OGCG-NEXT:     ret void
// OGCG-NEXT: }
