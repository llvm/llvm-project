// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir  %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

void A(void) {
  void *ptr = &&LABEL_A;
  goto *ptr;
LABEL_A:
  return;
}
// CIR:  cir.func {{.*}} @A
// CIR:    [[PTR:%.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["ptr", init] {alignment = 8 : i64}
// CIR:    [[BLOCK:%.*]] = cir.block_address <@A, "LABEL_A"> : !cir.ptr<!void>
// CIR:    cir.store align(8) [[BLOCK]], [[PTR]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    [[BLOCKADD:%.*]] = cir.load align(8) [[PTR]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:    cir.br ^bb1([[BLOCKADD]] : !cir.ptr<!void>)
// CIR:  ^bb1([[PHI:%.*]]: !cir.ptr<!void> {{.*}}):  // pred: ^bb0
// CIR:    cir.indirectbr [[PHI]] : <!void>, [
// CIR:    ^bb2
// CIR:    ]
// CIR:  ^bb2:  // pred: ^bb1
// CIR:    cir.label "LABEL_A"
// CIR:    cir.return

void B(void) {
LABEL_B:
  void *ptr = &&LABEL_B;
  goto *ptr;
}

// CIR:  cir.func {{.*}} @B()
// CIR:    [[PTR:%.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["ptr", init] {alignment = 8 : i64}
// CIR:    cir.br ^bb1
// CIR:   ^bb1: // 2 preds: ^bb0, ^bb2
// CIR:    cir.label "LABEL_B"
// CIR:    [[BLOCK:%.*]] = cir.block_address <@B, "LABEL_B"> : !cir.ptr<!void>
// CIR:    cir.store align(8) [[BLOCK]], [[PTR]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    [[BLOCKADD:%.*]] = cir.load align(8) [[PTR]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:    cir.br ^bb2([[BLOCKADD]] : !cir.ptr<!void>)
// CIR:  ^bb2([[PHI:%.*]]: !cir.ptr<!void> {{.*}}):  // pred: ^bb1
// CIR:    cir.indirectbr [[PHI]] : <!void>, [
// CIR-NEXT:    ^bb1
// CIR:    ]

void C(int x) {
  void *ptr = (x == 0) ? &&LABEL_A : &&LABEL_B;
  goto *ptr;
LABEL_A:
  return;
LABEL_B:
  return;
}

// CIR:  cir.func {{.*}} @C
// CIR:    [[BLOCK1:%.*]] = cir.block_address <@C, "LABEL_A"> : !cir.ptr<!void>
// CIR:    [[BLOCK2:%.*]] = cir.block_address <@C, "LABEL_B"> : !cir.ptr<!void>
// CIR:    [[COND:%.*]] = cir.select if [[CMP:%.*]] then [[BLOCK1]] else [[BLOCK2]] : (!cir.bool, !cir.ptr<!void>, !cir.ptr<!void>) -> !cir.ptr<!void>
// CIR:    cir.store align(8) [[COND]], [[PTR:%.*]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    [[BLOCKADD:%.*]] = cir.load align(8) [[PTR]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:    cir.br ^bb1([[BLOCKADD]] : !cir.ptr<!void>)
// CIR:  ^bb1([[PHI:%.*]]: !cir.ptr<!void> {{.*}}):  // pred: ^bb0
// CIR:    cir.indirectbr [[PHI]] : <!void>, [
// CIR-NEXT:    ^bb2,
// CIR-NEXT:    ^bb4
// CIR:    ]
// CIR:  ^bb2:  // pred: ^bb1
// CIR:    cir.label "LABEL_A"
// CIR:    cir.br ^bb3
// CIR:  ^bb3:  // 2 preds: ^bb2, ^bb4
// CIR:    cir.return
// CIR:  ^bb4:  // pred: ^bb1
// CIR:    cir.label "LABEL_B"
// CIR:    cir.br ^bb3

void D(void) {
  void *ptr = &&LABEL_A;
  void *ptr2 = &&LABEL_A;
  goto *ptr2;
LABEL_A:
  void *ptr3 = &&LABEL_A;
  return;
}

// CIR:  cir.func {{.*}} @D
// CIR:    %[[PTR:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["ptr", init]
// CIR:    %[[PTR2:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["ptr2", init]
// CIR:    %[[PTR3:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["ptr3", init]
// CIR:    %[[BLK1:.*]] = cir.block_address <@D, "LABEL_A"> : !cir.ptr<!void>
// CIR:    cir.store align(8) %[[BLK1]], %[[PTR]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    %[[BLK2:.*]] = cir.block_address <@D, "LABEL_A"> : !cir.ptr<!void>
// CIR:    cir.store align(8) %[[BLK2]], %[[PTR2]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    cir.br ^bb1
// CIR:  ^bb1([[PHI:%*.]]: !cir.ptr<!void> {{.*}}):  // pred: ^bb0
// CIR:    cir.indirectbr [[PHI]] : <!void>, [
// CIR-DAG:    ^bb2,
// CIR-DAG:    ^bb2,
// CIR-DAG:    ^bb2
// CIR:    ]
// CIR:  ^bb2:  // 3 preds: ^bb1, ^bb1, ^bb1
// CIR:    cir.label "LABEL_A"
// CIR:    %[[BLK3:.*]] = cir.block_address <@D, "LABEL_A"> : !cir.ptr<!void>
// CIR:    cir.store align(8) %[[BLK3]], %[[PTR3]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    cir.return


// This test checks that CIR preserves insertion order of blockaddresses
// for indirectbr, even if some were resolved immediately and others later.
void E(void) {
  void *ptr = &&LABEL_D;
  void *ptr2 = &&LABEL_C;
LABEL_A:
LABEL_B:
  void *ptr3 = &&LABEL_B;
  void *ptr4 = &&LABEL_A;
LABEL_C:
LABEL_D:
  return;
}

//CIR:  cir.func dso_local @E()
//CIR:  ^bb1({{.*}}: !cir.ptr<!void> {{.*}}):  // no predecessors
//CIR:    cir.indirectbr {{.*}} poison : <!void>, [
//CIR-NEXT:    ^bb5,
//CIR-NEXT:    ^bb4,
//CIR-NEXT:    ^bb3,
//CIR-NEXT:    ^bb2
//CIR:    ]
//CIR:  ^bb2:  // 2 preds: ^bb0, ^bb1
//CIR:    cir.label "LABEL_A"
//CIR:  ^bb3:  // 2 preds: ^bb1, ^bb2
//CIR:    cir.label "LABEL_B"
//CIR:  ^bb4:  // 2 preds: ^bb1, ^bb3
//CIR:    cir.label "LABEL_C"
//CIR:  ^bb5:  // 2 preds: ^bb1, ^bb4
//CIR:    cir.label "LABEL_D"
