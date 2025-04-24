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

// Verify null statement handling.
void f4(void) {
  ;
}

//      CIR: cir.func @f4()
// CIR-NEXT:   cir.return

//      LLVM: define void @f4()
// LLVM-NEXT:   ret void

//      OGCG: define{{.*}} void @f4()
// OGCG-NEXT: entry:
// OGCG-NEXT:   ret void

// Verify null statement as for-loop body.
void f5(void) {
  for (;;)
    ;
}

//      CIR: cir.func @f5()
// CIR-NEXT:   cir.scope {
// CIR-NEXT:      cir.for : cond {
// CIR-NEXT:        %0 = cir.const #true
// CIR-NEXT:        cir.condition(%0)
// CIR-NEXT:      } body {
// CIR-NEXT:        cir.yield
// CIR-NEXT:      } step {
// CIR-NEXT:        cir.yield
// CIR-NEXT:      }
// CIR-NEXT:   }
// CIR-NEXT:   cir.return
// CIR-NEXT: }

// LLVM: define void @f5()
// LLVM:   br label %[[SCOPE:.*]]
// LLVM: [[SCOPE]]:
// LLVM:   br label %[[LOOP:.*]]
// LLVM: [[LOOP]]:
// LLVM:   br i1 true, label %[[LOOP_STEP:.*]], label %[[LOOP_EXIT:.*]]
// LLVM: [[LOOP_STEP]]:
// LLVM:   br label %[[LOOP_BODY:.*]]
// LLVM: [[LOOP_BODY]]:
// LLVM:   br label %[[LOOP]]
// LLVM: [[LOOP_EXIT]]:
// LLVM:   ret void

// OGCG: define{{.*}} void @f5()
// OGCG: entry:
// OGCG:   br label %[[LOOP:.*]]
// OGCG: [[LOOP]]:
// OGCG:   br label %[[LOOP]]

int gv;
int f6(void) {
  return gv;
}

//      CIR: cir.func @f6() -> !s32i
// CIR-NEXT:   %[[RV:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CIR-NEXT:   %[[GV_PTR:.*]] = cir.get_global @gv : !cir.ptr<!s32i>
// CIR-NEXT:   %[[GV:.*]] = cir.load %[[GV_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   cir.store %[[GV]], %[[RV]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %[[R:.*]] = cir.load %[[RV]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   cir.return %[[R]] : !s32i

// LLVM:      define i32 @f6()
// LLVM-NEXT:   %[[RV_PTR:.*]] = alloca i32, i64 1, align 4
// LLVM-NEXT:   %[[GV:.*]] = load i32, ptr @gv, align 4
// LLVM-NEXT:   store i32 %[[GV]], ptr %[[RV_PTR]], align 4
// LLVM-NEXT:   %[[RV:.*]] = load i32, ptr %[[RV_PTR]], align 4
// LLVM-NEXT:   ret i32 %[[RV]]

// OGCG:      define{{.*}} i32 @f6()
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[GV:.*]] = load i32, ptr @gv, align 4
// OGCG-NEXT:   ret i32 %[[GV]]
