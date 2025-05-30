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
// CIR-NEXT:   cir.store{{.*}} %arg0, %[[I_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %[[I_IGNORED:.*]] = cir.load{{.*}} %[[I_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   %[[I:.*]] = cir.load{{.*}} %[[I_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   cir.store{{.*}} %[[I]], %[[RV]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %[[R:.*]] = cir.load{{.*}} %[[RV]] : !cir.ptr<!s32i>, !s32i
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
// CIR-NEXT:   cir.store{{.*}} %[[THREE]], %[[RV]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %[[R:.*]] = cir.load{{.*}} %0 : !cir.ptr<!s32i>, !s32i
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
// CIR-NEXT:   cir.store{{.*}} %[[THREE]], %[[I_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %[[I:.*]] = cir.load{{.*}} %[[I_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   cir.store{{.*}} %[[I]], %[[RV]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %[[R:.*]] = cir.load{{.*}} %[[RV]] : !cir.ptr<!s32i>, !s32i
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
// CIR-NEXT:   %[[GV:.*]] = cir.load{{.*}} %[[GV_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   cir.store{{.*}} %[[GV]], %[[RV]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %[[R:.*]] = cir.load{{.*}} %[[RV]] : !cir.ptr<!s32i>, !s32i
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

int f7(int a, int b, int c) {
  return a + (b + c);
}

// CIR: cir.func @f7
// CIR:  %[[A_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR:  %[[B_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CIR:  %[[C_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["c", init]
// CIR:  %[[A:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR:  %[[B:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR:  %[[C:.*]] = cir.load{{.*}} %[[C_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR:  %[[B_PLUS_C:.*]] = cir.binop(add, %[[B]], %[[C]]) nsw : !s32i
// CIR:  %[[RETVAL:.*]] = cir.binop(add, %[[A]], %[[B_PLUS_C]]) nsw : !s32i

// LLVM: define i32 @f7
// LLVM:   %[[A_PTR:.*]] = alloca i32, i64 1, align 4
// LLVM:   %[[B_PTR:.*]] = alloca i32, i64 1, align 4
// LLVM:   %[[C_PTR:.*]] = alloca i32, i64 1, align 4
// LLVM:   %[[A:.*]] = load i32, ptr %[[A_PTR]], align 4
// LLVM:   %[[B:.*]] = load i32, ptr %[[B_PTR]], align 4
// LLVM:   %[[C:.*]] = load i32, ptr %[[C_PTR]], align 4
// LLVM:   %[[B_PLUS_C:.*]] = add nsw i32 %[[B]], %[[C]]
// LLVM:   %[[RETVAL:.*]] = add nsw i32 %[[A]], %[[B_PLUS_C]]

// OGCG: define{{.*}} i32 @f7
// OGCG: entry:
// OGCG:   %[[A_PTR:.*]] = alloca i32, align 4
// OGCG:   %[[B_PTR:.*]] = alloca i32, align 4
// OGCG:   %[[C_PTR:.*]] = alloca i32, align 4
// OGCG:   %[[A:.*]] = load i32, ptr %[[A_PTR]], align 4
// OGCG:   %[[B:.*]] = load i32, ptr %[[B_PTR]], align 4
// OGCG:   %[[C:.*]] = load i32, ptr %[[C_PTR]], align 4
// OGCG:   %[[B_PLUS_C:.*]] = add nsw i32 %[[B]], %[[C]]
// OGCG:   %[[RETVAL:.*]] = add nsw i32 %[[A]], %[[B_PLUS_C]]

int f8(int *p) {
  (*p) = 2;
  return (*p);
}

// CIR: cir.func @f8
// CIR:    %[[P_PTR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["p", init]
// CIR:    %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR:    %[[P:.*]] = cir.load deref{{.*}} %[[P_PTR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:    cir.store{{.*}} %[[TWO]], %[[P]] : !s32i, !cir.ptr<!s32i>
// CIR:    %[[P2:.*]] = cir.load deref{{.*}} %[[P_PTR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:    %[[STAR_P:.*]] = cir.load{{.*}} %[[P2]] : !cir.ptr<!s32i>, !s32i

// LLVM: define i32 @f8
// LLVM:   %[[P_PTR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[P:.*]] = load ptr, ptr %[[P_PTR]], align 8
// LLVM:   store i32 2, ptr %[[P]], align 4
// LLVM:   %[[P2:.*]] = load ptr, ptr %[[P_PTR]], align 8
// LLVM:   %[[STAR_P:.*]] = load i32, ptr %[[P2]], align 4

// OGCG: define{{.*}} i32 @f8
// OGCG: entry:
// OGCG:   %[[P_PTR:.*]] = alloca ptr, align 8
// OGCG:   %[[P:.*]] = load ptr, ptr %[[P_PTR]], align 8
// OGCG:   store i32 2, ptr %[[P]], align 4
// OGCG:   %[[P2:.*]] = load ptr, ptr %[[P_PTR]], align 8
// OGCG:   %[[STAR_P:.*]] = load i32, ptr %[[P2]], align 4


void f9() {}

//      CIR: cir.func @f9()
// CIR-NEXT:   cir.return

//      LLVM: define void @f9()
// LLVM-NEXT:   ret void

//      OGCG: define{{.*}} void @f9()
// OGCG-NEXT: entry:
// OGCG-NEXT:   ret void

void f10(int arg0, ...) {}

//      CIR: cir.func @f10(%[[ARG0:.*]]: !s32i loc({{.*}}), ...)
// CIR-NEXT:   %[[ARG0_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["arg0", init] {alignment = 4 : i64}
// CIR-NEXT:   cir.store{{.*}} %[[ARG0]], %[[ARG0_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   cir.return

//      LLVM: define void @f10(i32 %[[ARG0:.*]], ...)
// LLVM-NEXT:   %[[ARG0_PTR:.*]] = alloca i32, i64 1, align 4
// LLVM-NEXT:   store i32 %[[ARG0]], ptr %[[ARG0_PTR]], align 4
// LLVM-NEXT:   ret void

//      OGCG: define{{.*}} void @f10(i32 noundef %[[ARG0:.*]], ...)
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[ARG0_PTR:.*]] = alloca i32, align 4
// OGCG-NEXT:   store i32 %[[ARG0]], ptr %[[ARG0_PTR]], align 4
// OGCG-NEXT:   ret void

typedef unsigned long size_type;
typedef unsigned long _Tp;

size_type max_size(void) {
  return (size_type)~0 / sizeof(_Tp);
}

// CIR: cir.func @max_size()
// CIR:   %0 = cir.alloca !u64i, !cir.ptr<!u64i>, ["__retval"] {alignment = 8 : i64}
// CIR:   %1 = cir.const #cir.int<0> : !s32i
// CIR:   %2 = cir.unary(not, %1) : !s32i, !s32i
// CIR:   %3 = cir.cast(integral, %2 : !s32i), !u64i
// CIR:   %4 = cir.const #cir.int<8> : !u64i
// CIR:   %5 = cir.binop(div, %3, %4) : !u64i

// LLVM: define i64 @max_size()
// LLVM:   store i64 2305843009213693951, ptr

// OGCG: define{{.*}} i64 @max_size()
// OGCG:   ret i64 2305843009213693951
// CHECK:   cir.store{{.*}} %5, %0 : !u64i, !cir.ptr<!u64i>
// CHECK:   %6 = cir.load{{.*}} %0 : !cir.ptr<!u64i>, !u64i
// CHECK:   cir.return %6 : !u64i
// CHECK:   }

enum A {
  A_one,
  A_two
};
enum A a;

// CHECK:   cir.global external @a = #cir.int<0> : !u32i

enum B : int;
enum B b;

// CHECK:   cir.global external @b = #cir.int<0> : !u32i


enum C : int {
  C_one,
  C_two
};
enum C c;

// CHECK:   cir.global external @c = #cir.int<0> : !u32i
