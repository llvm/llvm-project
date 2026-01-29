// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir  %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm  %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm  %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

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
// CIR:    cir.indirect_br [[PHI]] : !cir.ptr<!void>, [
// CIR:    ^bb2
// CIR:    ]
// CIR:  ^bb2:  // pred: ^bb1
// CIR:    cir.label "LABEL_A"
// CIR:    cir.return

// LLVM: define dso_local void @A()
// LLVM:   [[PTR:%.*]] = alloca ptr, i64 1, align 8
// LLVM:   store ptr blockaddress(@A, %[[LABEL_A:.*]]), ptr [[PTR]], align 8
// LLVM:   [[BLOCKADD:%.*]] = load ptr, ptr [[PTR]], align 8
// LLVM:   br label %[[indirectgoto:.*]]
// LLVM: [[indirectgoto]]:                                                ; preds = %[[ENTRY:.*]]
// LLVM:  [[PHI:%.*]] = phi ptr [ [[BLOCKADD]], %[[ENTRY]] ]
// LLVM:  indirectbr ptr [[PHI]], [label %[[LABEL_A]]]
// LLVM: [[LABEL_A]]:                                                ; preds = %[[indirectgoto]]
// LLVM:   ret void

// OGCG: define dso_local void @A()
// OGCG:   [[PTR:%.*]] = alloca ptr, align 8
// OGCG:   store ptr blockaddress(@A, %LABEL_A), ptr [[PTR]], align 8
// OGCG:   [[BLOCKADD:%.*]] = load ptr, ptr [[PTR]], align 8
// OGCG:   br label %indirectgoto
// OGCG: LABEL_A:                                                ; preds = %indirectgoto
// OGCG:   ret void
// OGCG: indirectgoto:                                     ; preds = %entry
// OGCG:   %indirect.goto.dest = phi ptr [ [[BLOCKADD]], %entry ]
// OGCG:   indirectbr ptr %indirect.goto.dest, [label %LABEL_A]

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
// CIR:    cir.indirect_br [[PHI]] : !cir.ptr<!void>, [
// CIR-NEXT:    ^bb1
// CIR:    ]

// LLVM: define dso_local void @B
// LLVM:   %[[PTR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   br label %[[LABEL_B:.*]]
// LLVM: [[LABEL_B]]:
// LLVM:   store ptr blockaddress(@B, %[[LABEL_B]]), ptr %[[PTR]], align 8
// LLVM:   [[BLOCKADD:%.*]] = load ptr, ptr %[[PTR]], align 8
// LLVM:   br label %[[indirectgoto:.*]]
// LLVM: [[indirectgoto]]:
// LLVM:   [[PHI:%.*]] = phi ptr [ [[BLOCKADD]], %[[LABEL_B]] ]
// LLVM:   indirectbr ptr [[PHI]], [label %[[LABEL_B]]]

// OGCG: define dso_local void @B
// OGCG:   [[PTR:%.*]] = alloca ptr, align 8
// OGCG:   br label %LABEL_B
// OGCG: LABEL_B:                                                ; preds = %indirectgoto, %entry
// OGCG:   store ptr blockaddress(@B, %LABEL_B), ptr [[PTR]], align 8
// OGCG:   [[BLOCKADD:%.*]] = load ptr, ptr [[PTR]], align 8
// OGCG:   br label %indirectgoto
// OGCG: indirectgoto:                                     ; preds = %LABEL_B
// OGCG:   %indirect.goto.dest = phi ptr [ [[BLOCKADD]], %LABEL_B ]
// OGCG:   indirectbr ptr %indirect.goto.dest, [label %LABEL_B]

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
// CIR:    cir.indirect_br [[PHI]] : !cir.ptr<!void>, [
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

// LLVM: define dso_local void @C(i32 %0)
// LLVM:   [[COND:%.*]] = select i1 [[CMP:%.*]], ptr blockaddress(@C, %[[LABEL_A:.*]]), ptr blockaddress(@C, %[[LABEL_B:.*]])
// LLVM:   store ptr [[COND]], ptr [[PTR:%.*]], align 8
// LLVM:   [[BLOCKADD:%.*]] = load ptr, ptr [[PTR]], align 8
// LLVM:   br label %[[indirectgoto:.*]]
// LLVM: [[indirectgoto]]:
// LLVM:   [[PHI:%.*]] = phi ptr [ [[BLOCKADD]], %[[ENTRY:.*]] ]
// LLVM:   indirectbr ptr [[PHI]], [label %[[LABEL_A]], label %[[LABEL_B]]]
// LLVM: [[LABEL_A]]:
// LLVM:   br label %[[RET:.*]]
// LLVM: [[RET]]:
// LLVM:   ret void
// LLVM: [[LABEL_B]]:
// LLVM:   br label %[[RET]]

// OGCG: define dso_local void @C
// OGCG:   [[COND:%.*]] = select i1 [[CMP:%.*]], ptr blockaddress(@C, %LABEL_A), ptr blockaddress(@C, %LABEL_B)
// OGCG:   store ptr [[COND]], ptr [[PTR:%.*]], align 8
// OGCG:   [[BLOCKADD:%.*]] = load ptr, ptr [[PTR]], align 8
// OGCG:   br label %indirectgoto
// OGCG: LABEL_A:                                                ; preds = %indirectgoto
// OGCG:   br label %return
// OGCG: LABEL_B:                                                ; preds = %indirectgoto
// OGCG:   br label %return
// OGCG: return:                                           ; preds = %LABEL_B, %LABEL_A
// OGCG:   ret void
// OGCG: indirectgoto:                                     ; preds = %entry
// OGCG:   %indirect.goto.dest = phi ptr [ [[BLOCKADD]], %entry ]
// OGCG:   indirectbr ptr %indirect.goto.dest, [label %LABEL_A, label %LABEL_B]

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
// CIR:    cir.indirect_br [[PHI]] : !cir.ptr<!void>, [
// CIR-DAG:    ^bb2,
// CIR-DAG:    ^bb2,
// CIR-DAG:    ^bb2
// CIR:    ]
// CIR:  ^bb2:  // 3 preds: ^bb1, ^bb1, ^bb1
// CIR:    cir.label "LABEL_A"
// CIR:    %[[BLK3:.*]] = cir.block_address <@D, "LABEL_A"> : !cir.ptr<!void>
// CIR:    cir.store align(8) %[[BLK3]], %[[PTR3]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    cir.return

// LLVM: define dso_local void @D
// LLVM:   %[[PTR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[PTR2:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[PTR3:.*]] = alloca ptr, i64 1, align 8
// LLVM:   store ptr blockaddress(@D, %[[LABEL_A:.*]]), ptr %[[PTR]], align 8
// LLVM:   store ptr blockaddress(@D, %[[LABEL_A]]), ptr %[[PTR2]], align 8
// LLVM:   %[[BLOCKADD:.*]] = load ptr, ptr %[[PTR2]], align 8
// LLVM:   br label %[[indirectgoto:.*]]
// LLVM: [[indirectgoto]]:
// LLVM:   [[PHI:%.*]] = phi ptr [ %[[BLOCKADD]], %[[ENTRY:.*]] ]
// LLVM:   indirectbr ptr [[PHI]], [label %[[LABEL_A]], label %[[LABEL_A]], label %[[LABEL_A]]]
// LLVM: [[LABEL_A]]:
// LLVM:   store ptr blockaddress(@D, %[[LABEL_A]]), ptr %[[PTR3]], align 8
// LLVM:   ret void

// OGCG: define dso_local void @D
// OGCG:   %[[PTR:.*]] = alloca ptr, align 8
// OGCG:   %[[PTR2:.*]] = alloca ptr, align 8
// OGCG:   %[[PTR3:.*]] = alloca ptr, align 8
// OGCG:   store ptr blockaddress(@D, %LABEL_A), ptr %[[PTR]], align 8
// OGCG:   store ptr blockaddress(@D, %LABEL_A), ptr %[[PTR2]], align 8
// OGCG:   %[[BLOCKADD:.*]] = load ptr, ptr %[[PTR2]], align 8
// OGCG:   br label %indirectgoto
// OGCG: LABEL_A:                                                ; preds = %indirectgoto, %indirectgoto, %indirectgoto
// OGCG:   store ptr blockaddress(@D, %LABEL_A), ptr %[[PTR3]], align 8
// OGCG:   ret void
// OGCG: indirectgoto:                                     ; preds = %entry
// OGCG:   %indirect.goto.dest = phi ptr [ %[[BLOCKADD]], %entry ]
// OGCG:   indirectbr ptr %indirect.goto.dest, [label %LABEL_A, label %LABEL_A, label %LABEL_A]

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

//CIR:  cir.func {{.*}} @E()
//CIR:  ^bb1({{.*}}: !cir.ptr<!void> {{.*}}):  // no predecessors
//CIR:    cir.indirect_br {{.*}} poison : !cir.ptr<!void>, [
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

// LLVM: define dso_local void @E()
// LLVM:   store ptr blockaddress(@E, %[[LABEL_D:.*]])
// LLVM:   store ptr blockaddress(@E, %[[LABEL_C:.*]])
// LLVM:   br label %[[LABEL_A:.*]]
// LLVM: [[indirectgoto:.*]]:                                                ; No predecessors!
// LLVM:   indirectbr ptr poison, [label %[[LABEL_D]], label %[[LABEL_C]], label %[[LABEL_B:.*]], label %[[LABEL_A]]]
// LLVM: [[LABEL_A]]:
// LLVM:   br label %[[LABEL_B]]
// LLVM: [[LABEL_B]]:
// LLVM:   store ptr blockaddress(@E, %[[LABEL_B]])
// LLVM:   store ptr blockaddress(@E, %[[LABEL_A]])
// LLVM:   br label %[[LABEL_C]]
// LLVM: [[LABEL_C]]:
// LLVM:   br label %[[LABEL_D]]
// LLVM: [[LABEL_D]]:

// OGCG: define dso_local void @E() #0 {
// OGCG:   store ptr blockaddress(@E, %LABEL_D), ptr %ptr, align 8
// OGCG:   store ptr blockaddress(@E, %LABEL_C), ptr %ptr2, align 8
// OGCG:   br label %LABEL_A
// OGCG: LABEL_A:                                                ; preds = %indirectgoto, %entry
// OGCG:   br label %LABEL_B
// OGCG: LABEL_B:                                                ; preds = %indirectgoto, %LABEL_A
// OGCG:   store ptr blockaddress(@E, %LABEL_B), ptr %ptr3, align 8
// OGCG:   store ptr blockaddress(@E, %LABEL_A), ptr %ptr4, align 8
// OGCG:   br label %LABEL_C
// OGCG: LABEL_C:                                                ; preds = %LABEL_B, %indirectgoto
// OGCG:   br label %LABEL_D
// OGCG: LABEL_D:                                                ; preds = %LABEL_C, %indirectgoto
// OGCG:   ret void
// OGCG: indirectgoto:                                     ; No predecessors!
// OGCG:   indirectbr ptr poison, [label %LABEL_D, label %LABEL_C, label %LABEL_B, label %LABEL_A]
