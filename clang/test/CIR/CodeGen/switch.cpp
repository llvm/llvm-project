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
// CIR: cir.func @sw1
// CIR: cir.switch (%3 : !s32i) {
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

// CIR: cir.func @sw2
// CIR: cir.scope {
// CIR-NEXT:   %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["yolo", init]
// CIR-NEXT:   %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["fomo", init]
// CIR:        cir.switch (%4 : !s32i) {
// CIR-NEXT:   cir.case(equal, [#cir.int<3> : !s32i]) {
// CIR-NEXT:     %5 = cir.const #cir.int<0> : !s32i
// CIR-NEXT:     cir.store %5, %2 : !s32i, !cir.ptr<!s32i>

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
