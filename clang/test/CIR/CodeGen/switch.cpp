// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
/// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
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
// CIR-NEXT:     cir.store %[[ZERO]], %[[FOMO]] : !s32i, !cir.ptr<!s32i>

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
// CIR-NEXT:   %[[COND:.*]] = cir.load %[[A:.*]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   cir.switch (%[[COND]] : !s32i) {
// CIR-NEXT:   cir.case(default, []) {
// CIR-NEXT:     cir.break
// CIR-NEXT:   }
// CIR-NEXT:   cir.yield
// CIR-NEXT:   }

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
// CIR-NEXT:           cir.store %[[THREE]], %[[RETVAL:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:           %[[RET3:.*]] = cir.load %[[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:           cir.return %[[RET3]] : !s32i
// CIR-NEXT:         }
// CIR-NEXT:         cir.yield
// CIR-NEXT:       }
// CIR-NEXT:       cir.case(default, []) {
// CIR-NEXT:         %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR-NEXT:         cir.store %[[TWO]], %[[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:         %[[RET2:.*]] = cir.load %[[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:         cir.return %[[RET2]] : !s32i
// CIR-NEXT:       }
// CIR-NEXT:       cir.yield
// CIR-NEXT:  }

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
// CIR-NEXT:   cir.store %[[ZERO]], %[[Y]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   cir.yield
// CIR-NEXT: }
// CIR-NEXT: cir.case(equal, [#cir.int<3> : !s32i]) {
// CIR-NEXT:   cir.break
// CIR-NEXT: }
// CIR-NEXT: cir.case(default, []) {
// CIR-NEXT:   cir.break
// CIR-NEXT: }

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
