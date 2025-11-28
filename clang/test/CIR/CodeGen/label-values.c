// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir  %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

void A(void) {
  void *ptr = &&LABEL_A;
LABEL_A:
  return;
}
// CIR:  cir.func dso_local @A
// CIR:    [[PTR:%.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["ptr", init] {alignment = 8 : i64}
// CIR:    [[BLOCK:%.*]] = cir.block_address <@A, "LABEL_A"> : !cir.ptr<!void>
// CIR:    cir.store align(8) [[BLOCK]], [[PTR]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    cir.br ^bb1
// CIR:  ^bb1:  // pred: ^bb0
// CIR:    cir.label "LABEL_A"
// CIR:    cir.return

void B(void) {
LABEL_B:
  void *ptr = &&LABEL_B;
}

// CIR:  cir.func dso_local @B()
// CIR:    [[PTR:%.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["ptr", init] {alignment = 8 : i64}
// CIR:    cir.br ^bb1
// CIR:   ^bb1:
// CIR:    cir.label "LABEL_B"
// CIR:    [[BLOCK:%.*]] = cir.block_address <@B, "LABEL_B"> : !cir.ptr<!void>
// CIR:    cir.store align(8) [[BLOCK]], [[PTR]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    cir.return

void C(int x) {
    void *ptr = (x == 0) ? &&LABEL_A : &&LABEL_B;
LABEL_A:
    return;
LABEL_B:
    return;
}

// CIR:  cir.func dso_local @C
// CIR:    [[BLOCK1:%.*]] = cir.block_address <@C, "LABEL_A"> : !cir.ptr<!void>
// CIR:    [[BLOCK2:%.*]] = cir.block_address <@C, "LABEL_B"> : !cir.ptr<!void>
// CIR:    [[COND:%.*]] = cir.select if [[CMP:%.*]] then [[BLOCK1]] else [[BLOCK2]] : (!cir.bool, !cir.ptr<!void>, !cir.ptr<!void>) -> !cir.ptr<!void>
// CIR:    cir.store align(8) [[COND]], [[PTR:%.*]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    cir.br ^bb1
// CIR:  ^bb1:  // pred: ^bb0
// CIR:    cir.label "LABEL_A"
// CIR:    cir.br ^bb2
// CIR:  ^bb2:  // 2 preds: ^bb1, ^bb3
// CIR:    cir.return
// CIR:  ^bb3:  // no predecessors
// CIR:    cir.label "LABEL_B"
// CIR:    cir.br ^bb2

void D(void) {
  void *ptr = &&LABEL_A;
  void *ptr2 = &&LABEL_A;
LABEL_A:
  void *ptr3 = &&LABEL_A;
  return;
}

// CIR:  cir.func dso_local @D
// CIR:    %[[PTR:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["ptr", init]
// CIR:    %[[PTR2:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["ptr2", init]
// CIR:    %[[PTR3:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["ptr3", init]
// CIR:    %[[BLK1:.*]] = cir.block_address <@D, "LABEL_A"> : !cir.ptr<!void>
// CIR:    cir.store align(8) %[[BLK1]], %[[PTR]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    %[[BLK2:.*]] = cir.block_address <@D, "LABEL_A"> : !cir.ptr<!void>
// CIR:    cir.store align(8) %[[BLK2]], %[[PTR2]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    cir.br ^bb1
// CIR:  ^bb1:  // pred: ^bb0
// CIR:    cir.label "LABEL_A"
// CIR:    %[[BLK3:.*]] = cir.block_address <@D, "LABEL_A"> : !cir.ptr<!void>
// CIR:    cir.store align(8) %[[BLK3]], %[[PTR3]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    cir.return
