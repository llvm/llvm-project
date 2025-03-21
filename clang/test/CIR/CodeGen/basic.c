// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

int f1(int i);

int f1(int i) {
  i;
  return i;
}

//      CIR: module
// CIR-NEXT: cir.func @f1(%arg0: !s32i loc({{.*}})) -> !s32i
// CIR-NEXT:   %[[I_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
// CIR-NEXT:   %[[RV:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CIR-NEXT:   cir.store %arg0, %[[I_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %[[I_IGNORED:.*]] = cir.load %[[I_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   %[[I:.*]] = cir.load %[[I_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   cir.store %[[I]], %[[RV]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %[[R:.*]] = cir.load %[[RV]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   cir.return %[[R]] : !s32i

//      LLVM: define i32 @f1(i32 %[[IP:.*]])
// LLVM-NEXT:   %[[I_PTR:.*]] = alloca i32, i64 1, align 4
// LLVM-NEXT:   %[[RV:.*]] = alloca i32, i64 1, align 4
// LLVM-NEXT:   store i32 %[[IP]], ptr %[[I_PTR]], align 4
// LLVM-NEXT:   %[[I_IGNORED:.*]] = load i32, ptr %[[I_PTR]], align 4
// LLVM-NEXT:   %[[I:.*]] = load i32, ptr %[[I_PTR]], align 4
// LLVM-NEXT:   store i32 %[[I]], ptr %[[RV]], align 4
// LLVM-NEXT:   %[[R:.*]] = load i32, ptr %[[RV]], align 4
// LLVM-NEXT:   ret i32 %[[R]]

//      OGCG: define{{.*}} i32 @f1(i32 noundef %[[I:.*]])
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[I_PTR:.*]] = alloca i32, align 4
// OGCG-NEXT:   store i32 %[[I]], ptr %[[I_PTR]], align 4
// OGCG-NEXT:   %[[I_IGNORED:.*]] = load i32, ptr %[[I_PTR]], align 4
// OGCG-NEXT:   %[[I:.*]] = load i32, ptr %[[I_PTR]], align 4
// OGCG-NEXT:   ret i32 %[[I]]

int f2(void) { return 3; }

//      CIR: cir.func @f2() -> !s32i
// CIR-NEXT:   %[[RV:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CIR-NEXT:   %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
// CIR-NEXT:   cir.store %[[THREE]], %[[RV]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %[[R:.*]] = cir.load %0 : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   cir.return %[[R]] : !s32i

//      LLVM: define i32 @f2()
// LLVM-NEXT:   %[[RV:.*]] = alloca i32, i64 1, align 4
// LLVM-NEXT:   store i32 3, ptr %[[RV]], align 4
// LLVM-NEXT:   %[[R:.*]] = load i32, ptr %[[RV]], align 4
// LLVM-NEXT:   ret i32 %[[R]]

//      OGCG: define{{.*}} i32 @f2()
// OGCG-NEXT: entry:
// OGCG-NEXT:   ret i32 3

int f3(void) {
  int i = 3;
  return i;
}

//      CIR: cir.func @f3() -> !s32i
// CIR-NEXT:   %[[RV:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CIR-NEXT:   %[[I_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
// CIR-NEXT:   %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
// CIR-NEXT:   cir.store %[[THREE]], %[[I_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %[[I:.*]] = cir.load %[[I_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   cir.store %[[I]], %[[RV]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %[[R:.*]] = cir.load %[[RV]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   cir.return %[[R]] : !s32i

//      LLVM: define i32 @f3()
// LLVM-NEXT:   %[[RV:.*]] = alloca i32, i64 1, align 4
// LLVM-NEXT:   %[[I_PTR:.*]] = alloca i32, i64 1, align 4
// LLVM-NEXT:   store i32 3, ptr %[[I_PTR]], align 4
// LLVM-NEXT:   %[[I:.*]] = load i32, ptr %[[I_PTR]], align 4
// LLVM-NEXT:   store i32 %[[I]], ptr %[[RV]], align 4
// LLVM-NEXT:   %[[R:.*]] = load i32, ptr %[[RV]], align 4
// LLVM-NEXT:   ret i32 %[[R]]

//      OGCG: define{{.*}} i32 @f3
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[I_PTR:.*]] = alloca i32, align 4
// OGCG-NEXT:   store i32 3, ptr %[[I_PTR]], align 4
// OGCG-NEXT:   %[[I:.*]] = load i32, ptr %[[I_PTR]], align 4
// OGCG-NEXT:   ret i32 %[[I]]
