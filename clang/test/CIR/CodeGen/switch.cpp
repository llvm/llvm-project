// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
/// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG
void sw1(int a) {
  switch (int b = 1; a) {
  case 0:
    b = b + 1;
    break;
  case 1:
    break;
  case 2: {
    b = b + 1;
    int yolo = 100;
    break;
  }
  }
}

// CIR: cir.func @_Z3sw1i
// CIR: cir.switch (%[[COND:.*]] : !s32i) {
// CIR-NEXT: cir.case(equal, [#cir.int<0> : !s32i]) {
// CIR: cir.break
// CIR: cir.case(equal, [#cir.int<1> : !s32i]) {
// CIR-NEXT: cir.break
// CIR: cir.case(equal, [#cir.int<2> : !s32i]) {
// CIR: cir.scope {
// CIR: cir.alloca !s32i, !cir.ptr<!s32i>, ["yolo", init]
// CIR: cir.break

// LLVM: define void @_Z3sw1i
// LLVM:   store i32 1, ptr %[[B_ADDR:.*]], align 4
// LLVM:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR:.*]], align 4
// LLVM:   br label %[[BB7:.*]]
// LLVM: [[BB7]]:
// LLVM:   switch i32 %[[A_VAL]], label %[[EXIT:.*]] [
// LLVM-DAG:   i32 0, label %[[CASE0:.*]]
// LLVM-DAG:   i32 1, label %[[CASE1:.*]]
// LLVM-DAG:   i32 2, label %[[CASE2:.*]]
// LLVM:   ]
// LLVM: [[CASE0]]:
// LLVM:   %[[B:.*]] = load i32, ptr %[[B_ADDR]], align 4
// LLVM:   %[[INC0:.*]] = add nsw i32 %[[B]], 1
// LLVM:   store i32 %[[INC0]], ptr %[[B_ADDR]], align 4
// LLVM:   br label %[[EXIT]]
// LLVM: [[CASE1]]:
// LLVM:   br label %[[EXIT]]
// LLVM: [[CASE2]]:
// LLVM:   br label %[[BB14:.*]]
// LLVM: [[BB14]]:
// LLVM:   %[[B2:.*]] = load i32, ptr %[[B_ADDR]], align 4
// LLVM:   %[[INC2:.*]] = add nsw i32 %[[B2]], 1
// LLVM:   store i32 %[[INC2]], ptr %[[B_ADDR]], align 4
// LLVM:   store i32 100, ptr %[[YOLO:.*]], align 4
// LLVM:   br label %[[EXIT]]
// LLVM: [[EXIT]]:
// LLVM:   br label %[[DEFAULT:.*]]
// LLVM: [[DEFAULT]]:
// LLVM:   ret void

// OGCG: define dso_local void @_Z3sw1i
// OGCG: entry:
// OGCG:   %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[B:.*]] = alloca i32, align 4
// OGCG:   %[[YOLO:.*]] = alloca i32, align 4
// OGCG:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG:   switch i32 %[[A_VAL]], label %[[SW_EPILOG:.*]] [
// OGCG:     i32 0, label %[[SW0:.*]]
// OGCG:     i32 1, label %[[SW1:.*]]
// OGCG:     i32 2, label %[[SW2:.*]]
// OGCG:   ]
// OGCG: [[SW0]]:
// OGCG:   %[[B_LOAD0:.*]] = load i32, ptr %[[B]], align 4
// OGCG:   %[[B_INC0:.*]] = add nsw i32 %[[B_LOAD0]], 1
// OGCG:   store i32 %[[B_INC0]], ptr %[[B]], align 4
// OGCG:   br label %[[SW_EPILOG]]
// OGCG: [[SW1]]:
// OGCG:   br label %[[SW_EPILOG]]
// OGCG: [[SW2]]:
// OGCG:   %[[B_LOAD2:.*]] = load i32, ptr %[[B]], align 4
// OGCG:   %[[B_INC2:.*]] = add nsw i32 %[[B_LOAD2]], 1
// OGCG:   store i32 %[[B_INC2]], ptr %[[B]], align 4
// OGCG:   store i32 100, ptr %[[YOLO]], align 4
// OGCG:   br label %[[SW_EPILOG]]
// OGCG: [[SW_EPILOG]]:
// OGCG:   ret void

void sw2(int a) {
  switch (int yolo = 2; a) {
  case 3:
    // "fomo" has the same lifetime as "yolo"
    int fomo = 0;
    yolo = yolo + fomo;
    break;
  }
}

// CIR: cir.func @_Z3sw2i
// CIR: cir.scope {
// CIR-NEXT:   %[[YOLO:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["yolo", init]
// CIR-NEXT:   %[[FOMO:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["fomo", init]
// CIR:        cir.switch (%[[COND:.*]] : !s32i) {
// CIR-NEXT:   cir.case(equal, [#cir.int<3> : !s32i]) {
// CIR-NEXT:     %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT:     cir.store{{.*}} %[[ZERO]], %[[FOMO]] : !s32i, !cir.ptr<!s32i>

// LLVM: define void @_Z3sw2i
// LLVM:   store i32 2, ptr %[[YOLO_ADDR:.*]], align 4
// LLVM:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR:.*]], align 4
// LLVM:   br label %[[SWITCH:.*]]
// LLVM: [[SWITCH]]:
// LLVM:   switch i32 %[[A_VAL]], label %[[EXIT:.*]] [
// LLVM:     i32 3, label %[[CASE3:.*]]
// LLVM:   ]
// LLVM: [[CASE3]]:
// LLVM:   store i32 0, ptr %[[FOMO_ADDR:.*]], align 4
// LLVM:   %[[YOLO_VAL:.*]] = load i32, ptr %[[YOLO_ADDR]], align 4
// LLVM:   %[[FOMO_VAL:.*]] = load i32, ptr %[[FOMO_ADDR]], align 4
// LLVM:   %[[YOLO_PLUS_FOMO:.*]] = add nsw i32 %[[YOLO_VAL]], %[[FOMO_VAL]]
// LLVM:   store i32 %[[YOLO_PLUS_FOMO]], ptr %[[YOLO_ADDR]], align 4
// LLVM:   br label %[[EXIT]]
// LLVM: [[EXIT]]:
// LLVM:   br label %[[RET:.*]]
// LLVM: [[RET]]:
// LLVM:   ret void

// OGCG: define dso_local void @_Z3sw2i
// OGCG: entry:
// OGCG:   %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[YOLO:.*]] = alloca i32, align 4
// OGCG:   %[[FOMO:.*]] = alloca i32, align 4
// OGCG:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG:   switch i32 %[[A_VAL]], label %[[SW_EPILOG:.*]] [
// OGCG:     i32 3, label %[[SW3:.*]]
// OGCG:   ]
// OGCG: [[SW3]]:
// OGCG:   %[[Y:.*]] = load i32, ptr %[[YOLO]], align 4
// OGCG:   %[[F:.*]] = load i32, ptr %[[FOMO]], align 4
// OGCG:   %[[SUM:.*]] = add nsw i32 %[[Y]], %[[F]]
// OGCG:   store i32 %[[SUM]], ptr %[[YOLO]], align 4
// OGCG:   br label %[[SW_EPILOG]]
// OGCG: [[SW_EPILOG]]:
// OGCG:   ret void

void sw3(int a) {
  switch (a) {
  default:
    break;
  }
}

// CIR: cir.func @_Z3sw3i
// CIR: cir.scope {
// CIR-NEXT:   %[[COND:.*]] = cir.load{{.*}} %[[A:.*]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   cir.switch (%[[COND]] : !s32i) {
// CIR-NEXT:   cir.case(default, []) {
// CIR-NEXT:     cir.break
// CIR-NEXT:   }
// CIR-NEXT:   cir.yield
// CIR-NEXT:   }

// LLVM-LABEL: define void @_Z3sw3i
// LLVM:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR:.*]], align 4
// LLVM:   br label %[[SWITCH:.*]]
// LLVM: [[SWITCH]]:
// LLVM:   switch i32 %[[A_VAL]], label %[[DEFAULT:.*]] [
// LLVM:   ]
// LLVM: [[DEFAULT]]:
// LLVM:   br label %[[EXIT:.*]]
// LLVM: [[EXIT:.*]]:
// LLVM:   br label %[[RET:.*]]
// LLVM: [[RET]]:
// LLVM:   ret void

// OGCG: define dso_local void @_Z3sw3i
// OGCG: entry:
// OGCG:   %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG:   switch i32 %[[A_VAL]], label %[[DEFAULT:.*]] [
// OGCG: [[DEFAULT]]:
// OGCG:   br label %[[EPILOG:.*]]
// OGCG: [[EPILOG]]:
// OGCG:   ret void

int sw4(int a) {
  switch (a) {
  case 42: {
    return 3;
  }
  default:
    return 2;
  }
  return 0;
}

// CIR: cir.func @_Z3sw4i
// CIR:       cir.switch (%[[COND:.*]] : !s32i) {
// CIR-NEXT:       cir.case(equal, [#cir.int<42> : !s32i]) {
// CIR-NEXT:         cir.scope {
// CIR-NEXT:           %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
// CIR-NEXT:           cir.store{{.*}} %[[THREE]], %[[RETVAL:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:           %[[RET3:.*]] = cir.load{{.*}} %[[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:           cir.return %[[RET3]] : !s32i
// CIR-NEXT:         }
// CIR-NEXT:         cir.yield
// CIR-NEXT:       }
// CIR-NEXT:       cir.case(default, []) {
// CIR-NEXT:         %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR-NEXT:         cir.store{{.*}} %[[TWO]], %[[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:         %[[RET2:.*]] = cir.load{{.*}} %[[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:         cir.return %[[RET2]] : !s32i
// CIR-NEXT:       }
// CIR-NEXT:       cir.yield
// CIR-NEXT:  }

// LLVM: define i32 @_Z3sw4i
// LLVM:   %[[A_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM:   %[[RET_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM:   br label %[[ENTRY:.*]]
// LLVM: [[ENTRY]]:
// LLVM:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVM:   br label %[[SWITCH:.*]]
// LLVM: [[SWITCH]]:
// LLVM:   switch i32 %[[A_VAL]], label %[[DEFAULT:.*]] [
// LLVM-DAG:     i32 42, label %[[CASE42:.*]]
// LLVM:   ]
// LLVM: [[CASE42]]:
// LLVM:   br label %[[CASE42_BODY:.*]]
// LLVM: [[CASE42_BODY]]:
// LLVM:   store i32 3, ptr %[[RET_ADDR]], align 4
// LLVM:   %[[RET3:.*]] = load i32, ptr %[[RET_ADDR]], align 4
// LLVM:   ret i32 %[[RET3]]
// LLVM: [[DEFAULT]]:
// LLVM:   store i32 2, ptr %[[RET_ADDR]], align 4
// LLVM:   %[[RET2:.*]] = load i32, ptr %[[RET_ADDR]], align 4
// LLVM:   ret i32 %[[RET2]]
// LLVM: [[EXIT_UNRE:.*]]:
// LLVM:   store i32 0, ptr %[[RET_ADDR]], align 4
// LLVM:   %[[RET0:.*]] = load i32, ptr %[[RET_ADDR]], align 4
// LLVM:   ret i32 %[[RET0]]

// OGCG: define dso_local noundef i32 @_Z3sw4i
// OGCG: entry:
// OGCG:   %[[RETVAL:.*]] = alloca i32, align 4
// OGCG:   %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG:   switch i32 %[[A_VAL]], label %[[DEFAULT:.*]] [
// OGCG:     i32 42, label %[[SW42:.*]]
// OGCG:   ]
// OGCG: [[SW42]]:
// OGCG:   br label %[[RETURN:.*]]
// OGCG: [[DEFAULT]]:
// OGCG:   br label %[[RETURN]]
// OGCG: [[RETURN]]:
// OGCG:   %[[RETVAL_LOAD:.*]] = load i32, ptr %[[RETVAL]], align 4
// OGCG:   ret i32 %[[RETVAL_LOAD]]

void sw5(int a) {
  switch (a) {
  case 1:;
  }
}

// CIR: cir.func @_Z3sw5i
// CIR: cir.switch (%[[A:.*]] : !s32i) {
// CIR-NEXT:   cir.case(equal, [#cir.int<1> : !s32i]) {
// CIR-NEXT:     cir.yield
// CIR-NEXT:   }
// CIR-NEXT:   cir.yield
// CIR-NEXT:   }

// LLVM-LABEL: define void @_Z3sw5i
// LLVM:   %[[A_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM:   br label %[[ENTRY:.*]]
// LLVM: [[ENTRY]]:
// LLVM:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVM:   br label %[[SWITCH:.*]]
// LLVM: [[SWITCH]]:
// LLVM:   switch i32 %[[A_VAL]], label %[[EXIT:.*]] [
// LLVM-DAG:     i32 1, label %[[CASE1:.*]]
// LLVM:   ]
// LLVM: [[CASE1]]:
// LLVM:   br label %[[EXIT:.*]]
// LLVM: [[EXIT]]:
// LLVM:   br label %[[RET:.*]]
// LLVM: [[RET]]:
// LLVM:   ret void

// OGCG: define dso_local void @_Z3sw5i
// OGCG: entry:
// OGCG:   %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG:   switch i32 %[[A_VAL]], label %[[SW_EPILOG:.*]] [
// OGCG:     i32 1, label %[[SW1:.*]]
// OGCG:   ]
// OGCG: [[SW1]]:
// OGCG:   br label %[[SW_EPILOG]]
// OGCG: [[SW_EPILOG]]:
// OGCG:   ret void

void sw6(int a) {
  switch (a) {
  case 0:
  case 1:
  case 2:
    break;
  case 3:
  case 4:
  case 5:
    break;
  }
}

// CIR: cir.func @_Z3sw6i
// CIR: cir.switch (%[[A:.*]] : !s32i) {
// CIR-NEXT: cir.case(equal, [#cir.int<0> : !s32i]) {
// CIR-NEXT:     cir.yield
// CIR-NEXT: }
// CIR-NEXT: cir.case(equal, [#cir.int<1> : !s32i]) {
// CIR-NEXT:     cir.yield
// CIR-NEXT: }
// CIR-NEXT: cir.case(equal, [#cir.int<2> : !s32i]) {
// CIR-NEXT:     cir.break
// CIR-NEXT: }
// CIR-NEXT: cir.case(equal, [#cir.int<3> : !s32i]) {
// CIR-NEXT:     cir.yield
// CIR-NEXT: }
// CIR-NEXT: cir.case(equal, [#cir.int<4> : !s32i]) {
// CIR-NEXT:     cir.yield
// CIR-NEXT: }
// CIR-NEXT: cir.case(equal, [#cir.int<5> : !s32i]) {
// CIR-NEXT:     cir.break
// CIR-NEXT: }

// LLVM: define void @_Z3sw6i
// LLVM:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR:.*]], align 4
// LLVM:   br label %[[SWITCH:.*]]
// LLVM: [[SWITCH]]:
// LLVM:   switch i32 %[[A_VAL]], label %[[EXIT:.*]] [
// LLVM-DAG:     i32 0, label %[[CASE0:.*]]
// LLVM-DAG:     i32 1, label %[[CASE1:.*]]
// LLVM-DAG:     i32 2, label %[[CASE2:.*]]
// LLVM-DAG:     i32 3, label %[[CASE3:.*]]
// LLVM-DAG:     i32 4, label %[[CASE4:.*]]
// LLVM-DAG:     i32 5, label %[[CASE5:.*]]
// LLVM:   ]
// LLVM: [[CASE0]]:
// LLVM:   br label %[[CASE0_CONT:.*]]
// LLVM: [[CASE0_CONT]]:
// LLVM:   br label %[[CASE1]]
// LLVM: [[CASE1]]:
// LLVM:   br label %[[CASE1_CONT:.*]]
// LLVM: [[CASE1_CONT]]:
// LLVM:   br label %[[CASE2]]
// LLVM: [[CASE2]]:
// LLVM:   br label %[[EXIT:.*]]
// LLVM: [[CASE3]]:
// LLVM:   br label %[[CASE3_CONT:.*]]
// LLVM: [[CASE3_CONT]]:
// LLVM:   br label %[[CASE4]]
// LLVM: [[CASE4]]:
// LLVM:   br label %[[CASE4_CONT:.*]]
// LLVM: [[CASE4_CONT]]:
// LLVM:   br label %[[CASE5]]
// LLVM: [[CASE5]]:
// LLVM:   br label %[[EXIT]]
// LLVM: [[EXIT]]:
// LLVM:   br label %[[RET:.*]]
// LLVM: [[RET]]:
// LLVM:   ret void

// OGCG: define dso_local void @_Z3sw6i
// OGCG: entry:
// OGCG:   %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG:   store i32 %a, ptr %[[A_ADDR]], align 4
// OGCG:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG:   switch i32 %[[A_VAL]], label %[[EPILOG:.*]] [
// OGCG:     i32 0, label %[[BB0:.*]]
// OGCG:     i32 1, label %[[BB0]]
// OGCG:     i32 2, label %[[BB0]]
// OGCG:     i32 3, label %[[BB1:.*]]
// OGCG:     i32 4, label %[[BB1]]
// OGCG:     i32 5, label %[[BB1]]
// OGCG:   ]
// OGCG: [[BB0]]:
// OGCG:   br label %[[EPILOG]]
// OGCG: [[BB1]]:
// OGCG:   br label %[[EPILOG]]
// OGCG: [[EPILOG]]:
// OGCG:   ret void

void sw7(int a) {
  switch (a) {
  case 0:
  case 1:
  case 2:
    int x;
  case 3:
  case 4:
  case 5:
    break;
  }
}

// CIR: cir.func @_Z3sw7i
// CIR: %[[X:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x"]
// CIR: cir.switch (%[[A:.*]] : !s32i)
// CIR-NEXT: cir.case(equal, [#cir.int<0> : !s32i]) {
// CIR-NEXT:     cir.yield
// CIR-NEXT: }
// CIR-NEXT: cir.case(equal, [#cir.int<1> : !s32i]) {
// CIR-NEXT:     cir.yield
// CIR-NEXT: }
// CIR-NEXT: cir.case(equal, [#cir.int<2> : !s32i]) {
// CIR-NEXT:     cir.yield
// CIR-NEXT: }
// CIR-NEXT: cir.case(equal, [#cir.int<3> : !s32i]) {
// CIR-NEXT:     cir.yield
// CIR-NEXT: }
// CIR-NEXT: cir.case(equal, [#cir.int<4> : !s32i]) {
// CIR-NEXT:     cir.yield
// CIR-NEXT: }
// CIR-NEXT: cir.case(equal, [#cir.int<5> : !s32i]) {
// CIR-NEXT:     cir.break
// CIR-NEXT: }
// CIR-NEXT: cir.yield
// CIR: }

// LLVM: define void @_Z3sw7i
// LLVM:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR:.*]], align 4
// LLVM:   br label %[[SWITCH:.*]]
// LLVM: [[SWITCH]]:
// LLVM:   switch i32 %[[A_VAL]], label %[[EXIT:.*]] [
// LLVM-DAG:     i32 0, label %[[CASE0:.*]]
// LLVM-DAG:     i32 1, label %[[CASE1:.*]]
// LLVM-DAG:     i32 2, label %[[CASE2:.*]]
// LLVM-DAG:     i32 3, label %[[CASE3:.*]]
// LLVM-DAG:     i32 4, label %[[CASE4:.*]]
// LLVM-DAG:     i32 5, label %[[CASE5:.*]]
// LLVM:   ]
// LLVM: [[CASE0]]:
// LLVM:   br label %[[CASE0_CONT:.*]]
// LLVM: [[CASE0_CONT]]:
// LLVM:   br label %[[CASE1]]
// LLVM: [[CASE1]]:
// LLVM:   br label %[[CASE1_CONT:.*]]
// LLVM: [[CASE1_CONT]]:
// LLVM:   br label %[[CASE2]]
// LLVM: [[CASE2]]:
// LLVM:   br label %[[CASE2_CONT:.*]]
// LLVM: [[CASE2_CONT]]:
// LLVM:   br label %[[CASE3]]
// LLVM: [[CASE3]]:
// LLVM:   br label %[[CASE3_CONT:.*]]
// LLVM: [[CASE3_CONT]]:
// LLVM:   br label %[[CASE4]]
// LLVM: [[CASE4]]:
// LLVM:   br label %[[CASE4_CONT:.*]]
// LLVM: [[CASE4_CONT]]:
// LLVM:   br label %[[CASE5]]
// LLVM: [[CASE5]]:
// LLVM:   br label %[[EXIT]]
// LLVM: [[EXIT]]:
// LLVM:   br label %[[RET:.*]]
// LLVM: [[RET]]:
// LLVM:   ret void

// OGCG: define dso_local void @_Z3sw7i
// OGCG: entry:
// OGCG:   %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG:   switch i32 %[[A_VAL]], label %[[EPILOG:.*]] [
// OGCG:     i32 0, label %[[BB0:.*]]
// OGCG:     i32 1, label %[[BB0]]
// OGCG:     i32 2, label %[[BB0]]
// OGCG:     i32 3, label %[[BB1:.*]]
// OGCG:     i32 4, label %[[BB1]]
// OGCG:     i32 5, label %[[BB1]]
// OGCG:   ]
// OGCG: [[BB0]]:
// OGCG:   br label %[[BB1]]
// OGCG: [[BB1]]:
// OGCG:   br label %[[EPILOG]]
// OGCG: [[EPILOG]]:
// OGCG:   ret void


void sw8(int a) {
  switch (a)
  {
  case 3:
    break;
  case 4:
  default:
    break;
  }
}

// CIR:    cir.func @_Z3sw8i
// CIR:    cir.switch (%[[A:.*]] : !s32i)
// CIR-NEXT: cir.case(equal, [#cir.int<3> : !s32i]) {
// CIR-NEXT:   cir.break
// CIR-NEXT: }
// CIR-NEXT: cir.case(equal, [#cir.int<4> : !s32i]) {
// CIR-NEXT:   cir.yield
// CIR-NEXT: }
// CIR-NEXT: cir.case(default, []) {
// CIR-NEXT:   cir.break
// CIR-NEXT: }

// LLVM: define void @_Z3sw8i
// LLVM:   switch i32 %[[COND:.*]], label %[[DEFAULT:.*]] [
// LLVM-DAG:  i32 3, label %[[CASE3:.*]]
// LLVM-DAG:  i32 4, label %[[CASE4:.*]]
// LLVM:   ]
// LLVM: [[CASE3]]:
// LLVM:   br label %[[EXIT:.*]]
// LLVM: [[CASE4]]:
// LLVM:   br label %[[CASE4_CONT:.*]]
// LLVM: [[CASE4_CONT]]:
// LLVM:   br label %[[DEFAULT]]
// LLVM: [[DEFAULT]]:
// LLVM:   br label %[[EXIT:.*]]
// LLVM: [[EXIT]]:
// LLVM:    br label %[[RET:.*]]
// LLVM: [[RET]]:
// LLVM:   ret void

// OGCG: define dso_local void @_Z3sw8i
// OGCG: entry:
// OGCG:   %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG:   switch i32 %[[A_VAL]], label %[[DEFAULT:.*]] [
// OGCG:     i32 3, label %[[SW3:.*]]
// OGCG:     i32 4, label %[[SW4:.*]]
// OGCG:   ]
// OGCG: [[SW3]]:
// OGCG:   br label %[[EPILOG:.*]]
// OGCG: [[SW4]]:
// OGCG:   br label %[[DEFAULT]]
// OGCG: [[DEFAULT]]:
// OGCG:   br label %[[EPILOG]]
// OGCG: [[EPILOG]]:
// OGCG:   ret void

void sw9(int a) {
  switch (a)
  {
  case 3:
    break;
  default:
  case 4:
    break;
  }
}

// CIR:    cir.func @_Z3sw9i
// CIR:    cir.switch (%[[A:.*]] : !s32i)
// CIR-NEXT: cir.case(equal, [#cir.int<3> : !s32i]) {
// CIR-NEXT:   cir.break
// CIR-NEXT: }
// CIR-NEXT: cir.case(default, []) {
// CIR-NEXT:   cir.yield
// CIR-NEXT: }
// CIR-NEXT: cir.case(equal, [#cir.int<4> : !s32i]) {
// CIR-NEXT:   cir.break
// CIR-NEXT: }

// LLVM: define void @_Z3sw9i
// LLVM:   switch i32 %[[COND:.*]], label %[[DEFAULT:.*]] [
// LLVM-DAG:     i32 3, label %[[CASE3:.*]]
// LLVM-DAG:     i32 4, label %[[CASE4:.*]]
// LLVM:   ]
// LLVM: [[CASE3]]:
// LLVM:   br label %[[EXIT:.*]]
// LLVM: [[DEFAULT]]:
// LLVM:   br label %[[DEFAULT_CONT:.*]]
// LLVM: [[DEFAULT_CONT]]:
// LLVM:   br label %[[CASE4]]
// LLVM: [[CASE4]]:
// LLVM:   br label %[[EXIT:.*]]
// LLVM: [[EXIT]]:
// LLVM:   br label %[[RET:.*]]
// LLVM: [[RET]]:
// LLVM:   ret void

// OGCG: define dso_local void @_Z3sw9i
// OGCG: entry:
// OGCG:   %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG:   switch i32 %[[A_VAL]], label %[[DEFAULT:.*]] [
// OGCG:     i32 3, label %[[SW3:.*]]
// OGCG:     i32 4, label %[[SW4:.*]]
// OGCG:   ]
// OGCG: [[SW3]]:
// OGCG:   br label %[[EPILOG:.*]]
// OGCG: [[DEFAULT]]:
// OGCG:   br label %[[SW4]]
// OGCG: [[SW4]]:
// OGCG:   br label %[[EPILOG]]
// OGCG: [[EPILOG]]:
// OGCG:   ret void

void sw10(int a) {
  switch (a)
  {
  case 3:
    break;
  case 4:
  default:
  case 5:
    break;
  }
}

// CIR:    cir.func @_Z4sw10i
// CIR:    cir.switch (%[[A:.*]] : !s32i)
// CIR-NEXT: cir.case(equal, [#cir.int<3> : !s32i]) {
// CIR-NEXT:   cir.break
// CIR-NEXT: }
// CIR-NEXT: cir.case(equal, [#cir.int<4> : !s32i]) {
// CIR-NEXT:   cir.yield
// CIR-NEXT: }
// CIR-NEXT: cir.case(default, []) {
// CIR-NEXT:   cir.yield
// CIR-NEXT: }
// CIR-NEXT: cir.case(equal, [#cir.int<5> : !s32i]) {
// CIR-NEXT:   cir.break
// CIR-NEXT: }

// LLVM: define void @_Z4sw10i
// LLVM:   switch i32 %[[COND:.*]], label %[[DEFAULT:.*]] [
// LLVM-DAG:     i32 3, label %[[CASE_3:.*]]
// LLVM-DAG:     i32 4, label %[[CASE_4:.*]]
// LLVM-DAG:     i32 5, label %[[CASE_5:.*]]
// LLVM:   ]
// LLVM: [[CASE_3]]:
// LLVM:   br label %[[EXIT:.*]]
// LLVM: [[CASE_4]]:
// LLVM:   br label %[[CASE4_CONT:.*]]
// LLVM: [[CASE4_CONT]]:
// LLVM:   br label %[[DEFAULT]]
// LLVM: [[DEFAULT]]:
// LLVM:   br label %[[DEFAULT_CONT:.*]]
// LLVM: [[DEFAULT_CONT]]:
// LLVM:   br label %[[CASE_5]]
// LLVM: [[CASE_5]]:
// LLVM:   br label %[[EXIT]]
// LLVM: [[EXIT]]:
// LLVM:   br label %[[RET:.*]]
// LLVM: [[RET]]:
// LLVM:   ret void

// OGCG: define dso_local void @_Z4sw10i
// OGCG: entry:
// OGCG:   %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG:   switch i32 %[[A_VAL]], label %[[DEFAULT:.*]] [
// OGCG:     i32 3, label %[[BB3:.*]]
// OGCG:     i32 4, label %[[BB4:.*]]
// OGCG:     i32 5, label %[[BB5:.*]]
// OGCG:   ]
// OGCG: [[BB3]]:
// OGCG:   br label %[[EPILOG:.*]]
// OGCG: [[BB4]]:
// OGCG:   br label %[[DEFAULT]]
// OGCG: [[DEFAULT]]:
// OGCG:   br label %[[BB5]]
// OGCG: [[BB5]]:
// OGCG:   br label %[[EPILOG]]
// OGCG: [[EPILOG]]:
// OGCG:   ret void

void sw11(int a) {
  switch (a)
  {
  case 3:
    break;
  case 4:
  case 5:
  default:
  case 6:
  case 7:
    break;
  }
}

// CIR:    cir.func @_Z4sw11i
// CIR:    cir.switch (%[[A:.*]] : !s32i)
// CIR-NEXT: cir.case(equal, [#cir.int<3> : !s32i]) {
// CIR-NEXT:   cir.break
// CIR-NEXT: }
// CIR-NEXT: cir.case(equal, [#cir.int<4> : !s32i]) {
// CIR-NEXT:   cir.yield
// CIR-NEXT: }
// CIR-NEXT: cir.case(equal, [#cir.int<5> : !s32i]) {
// CIR-NEXT:   cir.yield
// CIR-NEXT: }
// CIR-NEXT: cir.case(default, []) {
// CIR-NEXT:   cir.yield
// CIR-NEXT: }
// CIR-NEXT: cir.case(equal, [#cir.int<6> : !s32i]) {
// CIR-NEXT:   cir.yield
// CIR-NEXT: }
// CIR-NEXT: cir.case(equal, [#cir.int<7> : !s32i]) {
// CIR-NEXT:   cir.break
// CIR-NEXT: }

// LLVM: define void @_Z4sw11i
// LLVM:   switch i32 %[[COND:.*]], label %[[DEFAULT:.*]] [
// LLVM-DAG:     i32 3, label %[[CASE_3:.*]]
// LLVM-DAG:     i32 4, label %[[CASE_4:.*]]
// LLVM-DAG:     i32 5, label %[[CASE_5:.*]]
// LLVM-DAG:     i32 6, label %[[CASE_6:.*]]
// LLVM-DAG:     i32 7, label %[[CASE_7:.*]]
// LLVM:   ]
// LLVM: [[CASE_3]]:
// LLVM:   br label %[[EXIT:.*]]
// LLVM: [[CASE_4]]:
// LLVM:   br label %[[CASE4_CONT:.*]]
// LLVM: [[CASE4_CONT]]:
// LLVM:   br label %[[CASE_5]]
// LLVM: [[CASE_5]]:
// LLVM:   br label %[[CASE5_CONT:.*]]
// LLVM: [[CASE5_CONT]]:
// LLVM:   br label %[[DEFAULT]]
// LLVM: [[DEFAULT]]:
// LLVM:   br label %[[DEFAULT_CONT:.*]]
// LLVM: [[DEFAULT_CONT]]:
// LLVM:   br label %[[CASE_6]]
// LLVM: [[CASE_6]]:
// LLVM:   br label %[[CASE6_CONT:.*]]
// LLVM: [[CASE6_CONT]]:
// LLVM:   br label %[[CASE_7]]
// LLVM: [[CASE_7]]:
// LLVM:   br label %[[EXIT]]
// LLVM: [[EXIT]]:
// LLVM:   br label %[[RET:.*]]
// LLVM: [[RET]]:
// LLVM:   ret void

// OGCG: define dso_local void @_Z4sw11i
// OGCG: entry:
// OGCG:   %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG:   switch i32 %[[A_VAL]], label %[[DEFAULT:.*]] [
// OGCG:     i32 3, label %[[BB3:.*]]
// OGCG:     i32 4, label %[[BB4:.*]]
// OGCG:     i32 5, label %[[BB4]]
// OGCG:     i32 6, label %[[BB6:.*]]
// OGCG:     i32 7, label %[[BB6]]
// OGCG:   ]
// OGCG: [[BB3]]:
// OGCG:   br label %[[EPILOG:.*]]
// OGCG: [[BB4]]:
// OGCG:   br label %[[DEFAULT]]
// OGCG: [[DEFAULT]]:
// OGCG:   br label %[[BB6]]
// OGCG: [[BB6]]:
// OGCG:   br label %[[EPILOG]]
// OGCG: [[EPILOG]]:
// OGCG:   ret void

void sw12(int a) {
  switch (a)
  {
  case 3:
    return;
    break;
  }
}

//      CIR: cir.func @_Z4sw12i
//      CIR:   cir.scope {
//      CIR:     cir.switch
// CIR-NEXT:     cir.case(equal, [#cir.int<3> : !s32i]) {
// CIR-NEXT:       cir.return
// CIR-NEXT:     ^bb1:  // no predecessors
// CIR-NEXT:       cir.break
// CIR-NEXT:     }

// LLVM: define void @_Z4sw12i
// LLVM:   switch i32 %[[COND:.*]], label %[[EXIT:.*]] [
// LLVM-DAG:     i32 3, label %[[CASE_3:.*]]
// LLVM:   ]
// LLVM: [[CASE_3]]:
// LLVM:   ret void
// LLVM: [[UNREACHABLE:.*]]:
// LLVM:   br label %[[EXIT:.*]]
// LLVM: [[EXIT]]:
// LLVM:   br label %[[RET:.*]]
// LLVM: [[RET]]:
// LLVM:   ret void

// OGCG: define dso_local void @_Z4sw12i
// OGCG: entry:
// OGCG:   %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG:   switch i32 %[[A_VAL]], label %[[SW_DEFAULT:.*]] [
// OGCG:     i32 3, label %[[SW3:.*]]
// OGCG:   ]
// OGCG: [[SW3]]:
// OGCG:   br label %[[SW_DEFAULT]]
// OGCG: [[SW_DEFAULT]]:
// OGCG:   ret void

void sw13(int a, int b) {
  switch (a) {
  case 1:
    switch (b) {
    case 2:
      break;
    }
  }
}

//      CIR:  cir.func @_Z4sw13ii
//      CIR:    cir.scope {
//      CIR:      cir.switch
// CIR-NEXT:      cir.case(equal, [#cir.int<1> : !s32i]) {
// CIR-NEXT:        cir.scope {
//      CIR:          cir.switch
// CIR-NEXT:          cir.case(equal, [#cir.int<2> : !s32i]) {
// CIR-NEXT:            cir.break
// CIR-NEXT:          }
// CIR-NEXT:          cir.yield
// CIR-NEXT:        }
// CIR-NEXT:      }
//      CIR:    cir.yield
//      CIR:    }
//      CIR:    cir.return

// LLVM: define void @_Z4sw13ii
// LLVM:   switch i32 %[[COND:.*]], label %[[OUTER_EXIT:.*]] [
// LLVM-DAG:     i32 1, label %[[CASE_A_1:.*]]
// LLVM:   ]
// LLVM: [[CASE_A_1]]:
// LLVM:   br label %[[LOAD_B:.*]]
// LLVM: [[LOAD_B]]:
// LLVM:   %[[B_VAL:.*]] = load i32, ptr %[[B_ADDR:.*]], align 4
// LLVM:   br label %[[INNER_SWITCH:.*]]
// LLVM: [[INNER_SWITCH]]:
// LLVM:   switch i32 %[[B_VAL]], label %[[INNER_EXIT:.*]] [
// LLVM-DAG:     i32 2, label %[[CASE_B_2:.*]]
// LLVM:   ]
// LLVM: [[CASE_B_2]]:
// LLVM:   br label %[[INNER_EXIT]]
// LLVM: [[INNER_EXIT]]:
// LLVM:   br label %[[INNER_EXIT_CONT:.*]]
// LLVM: [[INNER_EXIT_CONT]]:
// LLVM:   br label %[[MERGE:.*]]
// LLVM: [[MERGE]]:
// LLVM:   br label %[[OUTER_EXIT]]
// LLVM: [[OUTER_EXIT]]:
// LLVM:   br label %[[EXIT:.*]]
// LLVM: [[EXIT]]:
// LLVM:   ret void

// OGCG: define dso_local void @_Z4sw13ii
// OGCG: entry:
// OGCG:   %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[B_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG:   switch i32 %[[A_VAL]], label %[[EPILOG2:.*]] [
// OGCG:     i32 1, label %[[SW1:.*]]
// OGCG:   ]
// OGCG: [[SW1]]:
// OGCG:   %[[B_VAL:.*]] = load i32, ptr %[[B_ADDR]], align 4
// OGCG:   switch i32 %[[B_VAL]], label %[[EPILOG:.*]] [
// OGCG:     i32 2, label %[[SW12:.*]]
// OGCG:   ]
// OGCG: [[SW12]]:
// OGCG:   br label %[[EPILOG]]
// OGCG: [[EPILOG]]:
// OGCG:   br label %[[EPILOG2]]
// OGCG: [[EPILOG2]]:
// OGCG:   ret void

void sw14(int x) {
  switch (x) {
    case 1:
    case 2:
    case 3 ... 6:
    case 7:
      break;
    default:
      break;
  }
}

// CIR:      cir.func @_Z4sw14i
// CIR:      cir.switch
// CIR-NEXT: cir.case(equal, [#cir.int<1> : !s32i]) {
// CIR-NEXT:   cir.yield
// CIR-NEXT: }
// CIR-NEXT: cir.case(equal, [#cir.int<2> : !s32i]) {
// CIR-NEXT:   cir.yield
// CIR-NEXT: }
// CIR-NEXT: cir.case(range, [#cir.int<3> : !s32i, #cir.int<6> : !s32i]) {
// CIR-NEXT:   cir.yield
// CIR-NEXT: }
// CIR-NEXT: cir.case(equal, [#cir.int<7> : !s32i]) {
// CIR-NEXT:   cir.break
// CIR-NEXT: }
// CIR-NEXT: cir.case(default, []) {
// CIR-NEXT:   cir.break
// CIR-NEXT: }

// LLVM: define void @_Z4sw14i
// LLVM:   switch i32 %[[COND:.*]], label %[[DEFAULT:.*]] [
// LLVM-DAG:     i32 1, label %[[CASE1:.*]]
// LLVM-DAG:     i32 2, label %[[CASE2:.*]]
// LLVM-DAG:     i32 3, label %[[CASE3_TO_6:.*]]
// LLVM-DAG:     i32 4, label %[[CASE3_TO_6]]
// LLVM-DAG:     i32 5, label %[[CASE3_TO_6]]
// LLVM-DAG:     i32 6, label %[[CASE3_TO_6]]
// LLVM-DAG:     i32 7, label %[[CASE7:.*]]
// LLVM:   ]
// LLVM: [[CASE1]]:
// LLVM:   br label %[[AFTER1:.*]]
// LLVM: [[AFTER1]]:
// LLVM:   br label %[[CASE2]]
// LLVM: [[CASE2]]:
// LLVM:   br label %[[AFTER2:.*]]
// LLVM: [[AFTER2]]:
// LLVM:   br label %[[CASE3_TO_6]]
// LLVM: [[CASE3_TO_6]]:
// LLVM:   br label %[[AFTER3_6:.*]]
// LLVM: [[AFTER3_6]]:
// LLVM:   br label %[[CASE7]]
// LLVM: [[CASE7]]:
// LLVM:   br label %[[EXIT1:.*]]
// LLVM: [[DEFAULT]]:
// LLVM:   br label %[[EXIT1]]
// LLVM: [[EXIT1]]:
// LLVM:   br label %[[RET:.*]]
// LLVM: [[RET]]:
// LLVM:   ret void

// OGCG: define dso_local void @_Z4sw14i
// OGCG: entry:
// OGCG:   %[[X_ADDR:.*]] = alloca i32, align 4
// OGCG:   store i32 %x, ptr %[[X_ADDR]], align 4
// OGCG:   %[[X_VAL:.*]] = load i32, ptr %[[X_ADDR]], align 4
// OGCG:   switch i32 %[[X_VAL]], label %[[DEFAULT:.*]] [
// OGCG-DAG:     i32 1, label %[[BB1:.*]]
// OGCG-DAG:     i32 2, label %[[BB1]]
// OGCG-DAG:     i32 3, label %[[BB2:.*]]
// OGCG-DAG:     i32 4, label %[[BB2]]
// OGCG-DAG:     i32 5, label %[[BB2]]
// OGCG-DAG:     i32 6, label %[[BB2]]
// OGCG-DAG:     i32 7, label %[[BB3:.*]]
// OGCG:   ]
// OGCG: [[BB1]]:
// OGCG:   br label %[[BB2]]
// OGCG: [[BB2]]:
// OGCG:   br label %[[BB3]]
// OGCG: [[BB3]]:
// OGCG:   br label %[[EPILOG:.*]]
// OGCG: [[DEFAULT]]:
// OGCG:   br label %[[EPILOG]]
// OGCG: [[EPILOG]]:
// OGCG:   ret void

void sw15(int x) {
  int y;
  switch (x) {
    case 1:
    case 2:
      y = 0;
    case 3:
      break;
    default:
      break;
  }
}

// CIR:      cir.func @_Z4sw15i
// CIR:      %[[Y:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["y"]
// CIR:      cir.switch
// CIR-NEXT: cir.case(equal, [#cir.int<1> : !s32i]) {
// CIR-NEXT:   cir.yield
// CIR-NEXT: }
// CIR-NEXT: cir.case(equal, [#cir.int<2> : !s32i]) {
// CIR-NEXT:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT:   cir.store{{.*}} %[[ZERO]], %[[Y]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   cir.yield
// CIR-NEXT: }
// CIR-NEXT: cir.case(equal, [#cir.int<3> : !s32i]) {
// CIR-NEXT:   cir.break
// CIR-NEXT: }
// CIR-NEXT: cir.case(default, []) {
// CIR-NEXT:   cir.break
// CIR-NEXT: }

// LLVM: define void @_Z4sw15i
// LLVM:   switch i32 %[[COND:.*]], label %[[DEFAULT:.*]] [
// LLVM-DAG:     i32 1, label %[[CASE1:.*]]
// LLVM-DAG:     i32 2, label %[[CASE2:.*]]
// LLVM-DAG:     i32 3, label %[[CASE3:.*]]
// LLVM:   ]
// LLVM: [[CASE1]]:
// LLVM:   br label %[[CASE1_CONT:.*]]
// LLVM: [[CASE1_CONT]]:
// LLVM:   br label %[[CASE2]]
// LLVM: [[CASE2]]:
// LLVM:   store i32 0, ptr %[[Y_ADDR:.*]], align 4
// LLVM:   br label %[[CASE2_CONT:.*]]
// LLVM: [[CASE2_CONT]]:
// LLVM:   br label %[[CASE3]]
// LLVM: [[CASE3]]:
// LLVM:   br label %[[EXIT:.*]]
// LLVM: [[DEFAULT]]:
// LLVM:   br label %[[EXIT]]
// LLVM: [[EXIT]]:
// LLVM:   br label %[[RET:.*]]
// LLVM: [[RET]]:
// LLVM:   ret void

// OGCG: define dso_local void @_Z4sw15i
// OGCG: entry:
// OGCG:   %[[X_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[Y:.*]] = alloca i32, align 4
// OGCG:   store i32 %x, ptr %[[X_ADDR]], align 4
// OGCG:   %[[X_VAL:.*]] = load i32, ptr %[[X_ADDR]], align 4
// OGCG:   switch i32 %[[X_VAL]], label %[[DEFAULT:.*]] [
// OGCG-DAG:     i32 1, label %[[BB0:.*]]
// OGCG-DAG:     i32 2, label %[[BB0]]
// OGCG-DAG:     i32 3, label %[[BB1:.*]]
// OGCG:   ]
// OGCG: [[BB0]]:
// OGCG:   store i32 0, ptr %[[Y]], align 4
// OGCG:   br label %[[BB1]]
// OGCG: [[BB1]]:
// OGCG:   br label %[[EPILOG:.*]]
// OGCG: [[DEFAULT]]:
// OGCG:   br label %[[EPILOG]]
// OGCG: [[EPILOG]]:
// OGCG:   ret void

int nested_switch(int a) {
  switch (int b = 1; a) {
  case 0:
    b = b + 1;
  case 1:
    return b;
  case 2: {
    b = b + 1;
    if (a > 1000) {
        case 9:
          b = a + b;
    }
    if (a > 500) {
        case 7:
          return a + b;
    }
    break;
  }
  }

  return 0;
}

// CIR: cir.switch (%[[COND:.*]] : !s32i) {
// CIR:   cir.case(equal, [#cir.int<0> : !s32i]) {
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.case(equal, [#cir.int<1> : !s32i]) {
// CIR:     cir.return
// CIR:   }
// CIR:   cir.case(equal, [#cir.int<2> : !s32i]) {
// CIR:     cir.scope {
// CIR:     cir.scope {
// CIR:       cir.if
// CIR:         cir.case(equal, [#cir.int<9> : !s32i]) {
// CIR:         cir.yield
// CIR:     cir.scope {
// CIR:         cir.if
// CIR:           cir.case(equal, [#cir.int<7> : !s32i]) {
// CIR:           cir.return

// LLVM: define i32 @_Z13nested_switchi
// LLVM:   %[[B_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM:   %[[A_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM:   %[[RES_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM:   store i32 %[[ARG:.*]], ptr %[[A_ADDR]], align 4
// LLVM:   br label %[[ENTRY:.*]]
// LLVM: [[ENTRY]]:
// LLVM:   store i32 1, ptr %[[B_ADDR]], align 4
// LLVM:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVM:   br label %[[SWITCH:.*]]
// LLVM: [[SWITCH]]:
// LLVM:   switch i32 %[[A_VAL]], label %[[EXIT:.*]] [
// LLVM-DAG:     i32 0, label %[[CASE0:.*]]
// LLVM-DAG:     i32 1, label %[[CASE1:.*]]
// LLVM-DAG:     i32 2, label %[[CASE2:.*]]
// LLVM-DAG:     i32 9, label %[[CASE9:.*]]
// LLVM-DAG:     i32 7, label %[[CASE7:.*]]
// LLVM:   ]
// LLVM: [[CASE0]]:
// LLVM:   %[[B0:.*]] = load i32, ptr %[[B_ADDR]], align 4
// LLVM:   %[[B1:.*]] = add nsw i32 %[[B0]], 1
// LLVM:   store i32 %[[B1]], ptr %[[B_ADDR]], align 4
// LLVM:   br label %[[CASE0_CONT:.*]]
// LLVM: [[CASE0_CONT]]:
// LLVM:   br label %[[CASE1]]
// LLVM: [[CASE1]]:
// LLVM:   %[[B1a:.*]] = load i32, ptr %[[B_ADDR]], align 4
// LLVM:   store i32 %[[B1a]], ptr %[[RES_ADDR]], align 4
// LLVM:   %[[RET1:.*]] = load i32, ptr %[[RES_ADDR]], align 4
// LLVM:   ret i32 %[[RET1]]
// LLVM: [[CASE2]]:
// LLVM:   br label %[[CASE2_BODY:.*]]
// LLVM: [[CASE2_BODY]]:
// LLVM:   %[[B2:.*]] = load i32, ptr %[[B_ADDR]], align 4
// LLVM:   %[[B3:.*]] = add nsw i32 %[[B2]], 1
// LLVM:   store i32 %[[B3]], ptr %[[B_ADDR]], align 4
// LLVM:   br label %[[CASE2_CONT:.*]]
// LLVM: [[CASE9]]:
// LLVM:   %[[A9:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVM:   %[[B4:.*]] = load i32, ptr %[[B_ADDR]], align 4
// LLVM:   %[[SUM9:.*]] = add nsw i32 %[[A9]], %[[B4]]
// LLVM:   store i32 %[[SUM9]], ptr %[[B_ADDR]], align 4
// LLVM:   br label %[[CASE2_CONT1:.*]]
// LLVM: [[CASE7]]:
// LLVM:   %[[A7:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVM:   %[[B5:.*]] = load i32, ptr %[[B_ADDR]], align 4
// LLVM:   %[[SUM7:.*]] = add nsw i32 %[[A7]], %[[B5]]
// LLVM:   store i32 %[[SUM7]], ptr %[[RES_ADDR]], align 4
// LLVM:   %[[RET7:.*]] = load i32, ptr %[[RES_ADDR]], align 4
// LLVM:   ret i32 %[[RET7]]
// LLVM: [[EXIT]]:
// LLVM:   store i32 0, ptr %[[RES_ADDR]], align 4
// LLVM:   %[[RET0:.*]] = load i32, ptr %[[RES_ADDR]], align 4
// LLVM:   ret i32 %[[RET0]]

// OGCG: define dso_local noundef i32 @_Z13nested_switchi
// OGCG: entry:
// OGCG:   %[[RETVAL:.*]] = alloca i32, align 4
// OGCG:   %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[B:.*]] = alloca i32, align 4
// OGCG:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG:   switch i32 %[[A_VAL]], label %[[EPILOG:.*]] [
// OGCG:     i32 0, label %[[SW0:.*]]
// OGCG:     i32 1, label %[[SW1:.*]]
// OGCG:     i32 2, label %[[SW2:.*]]
// OGCG:     i32 9, label %[[SW4:.*]]
// OGCG:     i32 7, label %[[SW8:.*]]
// OGCG:   ]
// OGCG: [[SW0]]:
// OGCG:   %[[B_VAL0:.*]] = load i32, ptr %[[B]], align 4
// OGCG:   %[[ADD0:.*]] = add nsw i32 %[[B_VAL0]], 1
// OGCG:   br label %[[SW1]]
// OGCG: [[SW1]]:
// OGCG:   %[[B_VAL1:.*]] = load i32, ptr %[[B]], align 4
// OGCG:   br label %[[RETURN:.*]]
// OGCG: [[SW2]]:
// OGCG:   %[[B_VAL2:.*]] = load i32, ptr %[[B]], align 4
// OGCG:   %[[ADD2:.*]] = add nsw i32 %[[B_VAL2]], 1
// OGCG:   %[[A_VAL2:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG:   %[[CMP1000:.*]] = icmp sgt i32 %[[A_VAL2]], 1000
// OGCG:   br i1 %[[CMP1000]], label %[[IFTHEN:.*]], label %[[IFEND:.*]]
// OGCG: [[IFTHEN]]:
// OGCG:   br label %[[SW4]]
// OGCG: [[SW4]]:
// OGCG:   %[[A_VAL4:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG:   %[[B_VAL4:.*]] = load i32, ptr %[[B]], align 4
// OGCG:   %[[ADD4:.*]] = add nsw i32 %[[A_VAL4]], %[[B_VAL4]]
// OGCG:   br label %[[IFEND]]
// OGCG: [[IFEND]]:
// OGCG:   %[[A_VAL5:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG:   %[[CMP500:.*]] = icmp sgt i32 %[[A_VAL5]], 500
// OGCG:   br i1 %[[CMP500]], label %[[IFTHEN7:.*]], label %[[IFEND10:.*]]
// OGCG: [[IFTHEN7]]:
// OGCG:   br label %[[SW8]]
// OGCG: [[SW8]]:
// OGCG:   %[[A_VAL8:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG:   %[[B_VAL8:.*]] = load i32, ptr %[[B]], align 4
// OGCG:   %[[ADD8:.*]] = add nsw i32 %[[A_VAL8]], %[[B_VAL8]]
// OGCG:   br label %[[RETURN]]
// OGCG: [[IFEND10]]:
// OGCG:   br label %[[EPILOG]]
// OGCG: [[EPILOG]]:
