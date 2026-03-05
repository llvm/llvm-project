// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir  %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm  %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm  %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

void A(void) {
  void *ptr = &&A;
  goto *ptr;
A:
  return;
}
// CIR:  cir.func {{.*}} @A
// CIR:    [[PTR:%.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["ptr", init] {alignment = 8 : i64}
// CIR:    [[BLOCK:%.*]] = cir.blockaddress <@A, "A"> -> !cir.ptr<!void>
// CIR:    cir.store align(8) [[BLOCK]], [[PTR]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    [[BLOCKADD:%.*]] = cir.load align(8) [[PTR]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:    cir.br ^bb1([[BLOCKADD]] : !cir.ptr<!void>)
// CIR:  ^bb1([[PHI:%.*]]: !cir.ptr<!void> {{.*}}):  // pred: ^bb0
// CIR:    cir.indirectbr [[PHI]] : <!void>, [
// CIR:    ^bb2
// CIR:    ]
// CIR:  ^bb2:  // pred: ^bb1
// CIR:    cir.label "A"
// CIR:    cir.return
//
// LLVM: define dso_local void @A()
// LLVM:   [[PTR:%.*]] = alloca ptr, i64 1, align 8
// LLVM:   store ptr blockaddress(@A, %[[A:.*]]), ptr [[PTR]], align 8
// LLVM:   [[BLOCKADD:%.*]] = load ptr, ptr [[PTR]], align 8
// LLVM:   br label %[[indirectgoto:.*]]
// LLVM: [[indirectgoto]]:                                                ; preds = %[[ENTRY:.*]]
// LLVM:  [[PHI:%.*]] = phi ptr [ [[BLOCKADD]], %[[ENTRY]] ]
// LLVM:  indirectbr ptr [[PHI]], [label %[[A]]]
// LLVM: [[A]]:                                                ; preds = %[[indirectgoto]]
// LLVM:   ret void

// OGCG: define dso_local void @A()
// OGCG:   [[PTR:%.*]] = alloca ptr, align 8
// OGCG:   store ptr blockaddress(@A, %A), ptr [[PTR]], align 8
// OGCG:   [[BLOCKADD:%.*]] = load ptr, ptr [[PTR]], align 8
// OGCG:   br label %indirectgoto
// OGCG: A:                                                ; preds = %indirectgoto
// OGCG:   ret void
// OGCG: indirectgoto:                                     ; preds = %entry
// OGCG:   %indirect.goto.dest = phi ptr [ [[BLOCKADD]], %entry ]
// OGCG:   indirectbr ptr %indirect.goto.dest, [label %A]

void B(void) {
B:
  void *ptr = &&B;
  goto *ptr;
}

// CIR:  cir.func {{.*}} @B()
// CIR:    [[PTR:%.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["ptr", init] {alignment = 8 : i64}
// CIR:    cir.br ^bb1
// CIR:   ^bb1: // 2 preds: ^bb0, ^bb2
// CIR:    cir.label "B"
// CIR:    [[BLOCK:%.*]] = cir.blockaddress <@B, "B"> -> !cir.ptr<!void>
// CIR:    cir.store align(8) [[BLOCK]], [[PTR]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    [[BLOCKADD:%.*]] = cir.load align(8) [[PTR]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:    cir.br ^bb2([[BLOCKADD]] : !cir.ptr<!void>)
// CIR:  ^bb2([[PHI:%.*]]: !cir.ptr<!void> {{.*}}):  // pred: ^bb1
// CIR:    cir.indirectbr [[PHI]] : <!void>, [
// CIR-NEXT:    ^bb1
// CIR:    ]

// LLVM: define dso_local void @B
// LLVM:   %[[PTR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   br label %[[B:.*]]
// LLVM: [[B]]:
// LLVM:   store ptr blockaddress(@B, %[[B]]), ptr %[[PTR]], align 8
// LLVM:   [[BLOCKADD:%.*]] = load ptr, ptr %[[PTR]], align 8
// LLVM:   br label %[[indirectgoto:.*]]
// LLVM: [[indirectgoto]]:
// LLVM:   [[PHI:%.*]] = phi ptr [ [[BLOCKADD]], %[[B]] ]
// LLVM:   indirectbr ptr [[PHI]], [label %[[B]]]

// OGCG: define dso_local void @B
// OGCG:   [[PTR:%.*]] = alloca ptr, align 8
// OGCG:   br label %B
// OGCG: B:                                                ; preds = %indirectgoto, %entry
// OGCG:   store ptr blockaddress(@B, %B), ptr [[PTR]], align 8
// OGCG:   [[BLOCKADD:%.*]] = load ptr, ptr [[PTR]], align 8
// OGCG:   br label %indirectgoto
// OGCG: indirectgoto:                                     ; preds = %B
// OGCG:   %indirect.goto.dest = phi ptr [ [[BLOCKADD]], %B ]
// OGCG:   indirectbr ptr %indirect.goto.dest, [label %B]

void C(int x) {
  void *ptr = (x == 0) ? &&A : &&B;
  goto *ptr;
A:
    return;
B:
    return;
}

// CIR:  cir.func {{.*}} @C
// CIR:    [[BLOCK1:%.*]] = cir.blockaddress <@C, "A"> -> !cir.ptr<!void>
// CIR:    [[BLOCK2:%.*]] = cir.blockaddress <@C, "B"> -> !cir.ptr<!void>
// CIR:    [[COND:%.*]] = cir.select if [[CMP:%.*]] then [[BLOCK1]] else [[BLOCK2]] : (!cir.bool, !cir.ptr<!void>, !cir.ptr<!void>) -> !cir.ptr<!void>
// CIR:    cir.store align(8) [[COND]], [[PTR:%.*]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    [[BLOCKADD:%.*]] = cir.load align(8) [[PTR]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:    cir.br ^bb2([[BLOCKADD]] : !cir.ptr<!void>)
// CIR:  ^bb1:  // 2 preds: ^bb3, ^bb4
// CIR:    cir.return
// CIR:  ^bb2([[PHI:%.*]]: !cir.ptr<!void> {{.*}}):  // pred: ^bb0
// CIR:    cir.indirectbr [[PHI]] : <!void>, [
// CIR-NEXT:    ^bb3,
// CIR-NEXT:    ^bb4
// CIR:    ]
// CIR:  ^bb3:  // pred: ^bb2
// CIR:    cir.label "A"
// CIR:    cir.br ^bb1
// CIR:  ^bb4:  // pred: ^bb2
// CIR:    cir.label "B"
// CIR:    cir.br ^bb1

// LLVM: define dso_local void @C(i32 %0)
// LLVM:   [[COND:%.*]] = select i1 [[CMP:%.*]], ptr blockaddress(@C, %[[A:.*]]), ptr blockaddress(@C, %[[B:.*]])
// LLVM:   store ptr [[COND]], ptr [[PTR:%.*]], align 8
// LLVM:   [[BLOCKADD:%.*]] = load ptr, ptr [[PTR]], align 8
// LLVM:   br label %[[indirectgoto:.*]]
// LLVM: [[RET:.*]]:
// LLVM:   ret void
// LLVM: [[indirectgoto]]:
// LLVM:   [[PHI:%.*]] = phi ptr [ [[BLOCKADD]], %[[ENTRY:.*]] ]
// LLVM:   indirectbr ptr [[PHI]], [label %[[A]], label %[[B]]]
// LLVM: [[A]]:
// LLVM:   br label %[[RET]]
// LLVM: [[B]]:
// LLVM:   br label %[[RET]]

// OGCG: define dso_local void @C
// OGCG:   [[COND:%.*]] = select i1 [[CMP:%.*]], ptr blockaddress(@C, %A), ptr blockaddress(@C, %B)
// OGCG:   store ptr [[COND]], ptr [[PTR:%.*]], align 8
// OGCG:   [[BLOCKADD:%.*]] = load ptr, ptr [[PTR]], align 8
// OGCG:   br label %indirectgoto
// OGCG: A:                                                ; preds = %indirectgoto
// OGCG:   br label %return
// OGCG: B:                                                ; preds = %indirectgoto
// OGCG:   br label %return
// OGCG: return:                                           ; preds = %B, %A
// OGCG:   ret void
// OGCG: indirectgoto:                                     ; preds = %entry
// OGCG:   %indirect.goto.dest = phi ptr [ [[BLOCKADD]], %entry ]
// OGCG:   indirectbr ptr %indirect.goto.dest, [label %A, label %B]

void D(void) {
  void *ptr = &&A;
  void *ptr2 = &&A;
  goto *ptr2;
A:
  void *ptr3 = &&A;
  return;
}

// CIR:  cir.func {{.*}} @D
// CIR:    %[[PTR:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["ptr", init]
// CIR:    %[[PTR2:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["ptr2", init]
// CIR:    %[[PTR3:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["ptr3", init]
// CIR:    %[[BLK1:.*]] = cir.blockaddress <@D, "A"> -> !cir.ptr<!void>
// CIR:    cir.store align(8) %[[BLK1]], %[[PTR]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    %[[BLK2:.*]] = cir.blockaddress <@D, "A"> -> !cir.ptr<!void>
// CIR:    cir.store align(8) %[[BLK2]], %[[PTR2]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    %[[BLOCKADD:.*]] = cir.load align(8) %[[PTR2]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:    cir.br ^bb1(%[[BLOCKADD]] : !cir.ptr<!void>)
// CIR:  ^bb1([[PHI:%*.]]: !cir.ptr<!void> {{.*}}):  // pred: ^bb0
// CIR:    cir.indirectbr [[PHI]] : <!void>, [
// CIR-DAG:    ^bb2,
// CIR-DAG:    ^bb2,
// CIR-DAG:    ^bb2
// CIR:    ]
// CIR:  ^bb2:  // 3 preds: ^bb1, ^bb1, ^bb1
// CIR:    cir.label "A"
// CIR:    %[[BLK3:.*]] = cir.blockaddress <@D, "A"> -> !cir.ptr<!void>
// CIR:    cir.store align(8) %[[BLK3]], %[[PTR3]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    cir.return

// LLVM: define dso_local void @D
// LLVM:   %[[PTR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[PTR2:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[PTR3:.*]] = alloca ptr, i64 1, align 8
// LLVM:   store ptr blockaddress(@D, %[[A:.*]]), ptr %[[PTR]], align 8
// LLVM:   store ptr blockaddress(@D, %[[A]]), ptr %[[PTR2]], align 8
// LLVM:   %[[BLOCKADD:.*]] = load ptr, ptr %[[PTR2]], align 8
// LLVM:   br label %[[indirectgoto:.*]]
// LLVM: [[indirectgoto]]:
// LLVM:   [[PHI:%.*]] = phi ptr [ %[[BLOCKADD]], %[[ENTRY:.*]] ]
// LLVM:   indirectbr ptr [[PHI]], [label %[[A]], label %[[A]], label %[[A]]]
// LLVM: [[A]]:
// LLVM:   store ptr blockaddress(@D, %[[A]]), ptr %[[PTR3]], align 8
// LLVM:   ret void

// OGCG: define dso_local void @D
// OGCG:   %[[PTR:.*]] = alloca ptr, align 8
// OGCG:   %[[PTR2:.*]] = alloca ptr, align 8
// OGCG:   %[[PTR3:.*]] = alloca ptr, align 8
// OGCG:   store ptr blockaddress(@D, %A), ptr %[[PTR]], align 8
// OGCG:   store ptr blockaddress(@D, %A), ptr %[[PTR2]], align 8
// OGCG:   %[[BLOCKADD:.*]] = load ptr, ptr %[[PTR2]], align 8
// OGCG:   br label %indirectgoto
// OGCG: A:                                                ; preds = %indirectgoto, %indirectgoto, %indirectgoto
// OGCG:   store ptr blockaddress(@D, %A), ptr %[[PTR3]], align 8
// OGCG:   ret void
// OGCG: indirectgoto:                                     ; preds = %entry
// OGCG:   %indirect.goto.dest = phi ptr [ %[[BLOCKADD]], %entry ]
// OGCG:   indirectbr ptr %indirect.goto.dest, [label %A, label %A, label %A]

// This test checks that CIR preserves insertion order of blockaddresses
// for indirectbr, even if some were resolved immediately and others later.
void E(void) {
  void *ptr = &&D;
  void *ptr2 = &&C;
A:
B:
  void *ptr3 = &&B;
  void *ptr4 = &&A;
C:
D:
  return;
}

//CIR:  cir.func {{.*}} @E()
//CIR:  ^bb1({{.*}}: !cir.ptr<!void> {{.*}}):  // no predecessors
//CIR:    cir.indirectbr {{.*}} poison : <!void>, [
//CIR-NEXT:    ^bb5,
//CIR-NEXT:    ^bb4,
//CIR-NEXT:    ^bb3,
//CIR-NEXT:    ^bb2
//CIR:    ]
//CIR:  ^bb2:  // 2 preds: ^bb0, ^bb1
//CIR:    cir.label "A" loc(#loc65)
//CIR:  ^bb3:  // 2 preds: ^bb1, ^bb2
//CIR:    cir.label "B" loc(#loc66)
//CIR:  ^bb4:  // 2 preds: ^bb1, ^bb3
//CIR:    cir.label "C"
//CIR:  ^bb5:  // 2 preds: ^bb1, ^bb4
//CIR:    cir.label "D"

// LLVM: define dso_local void @E()
// LLVM:   store ptr blockaddress(@E, %[[D:.*]])
// LLVM:   store ptr blockaddress(@E, %[[C:.*]])
// LLVM:   br label %[[A:.*]]
// LLVM: [[indirectgoto:.*]]:                                                ; No predecessors!
// LLVM:   indirectbr ptr poison, [label %[[D]], label %[[C]], label %[[B:.*]], label %[[A]]]
// LLVM: [[A]]:
// LLVM:   br label %[[B]]
// LLVM: [[B]]:
// LLVM:   store ptr blockaddress(@E, %[[B]]), ptr %3, align 8
// LLVM:   store ptr blockaddress(@E, %[[A]]), ptr %4, align 8
// LLVM:   br label %8
// LLVM: [[C]]:
// LLVM:   br label %9
// LLVM: [[D]]:

// OGCG: define dso_local void @E() #0 {
// OGCG:   store ptr blockaddress(@E, %D), ptr %ptr, align 8
// OGCG:   store ptr blockaddress(@E, %C), ptr %ptr2, align 8
// OGCG:   br label %A
// OGCG: A:                                                ; preds = %indirectgoto, %entry
// OGCG:   br label %B
// OGCG: B:                                                ; preds = %indirectgoto, %A
// OGCG:   store ptr blockaddress(@E, %B), ptr %ptr3, align 8
// OGCG:   store ptr blockaddress(@E, %A), ptr %ptr4, align 8
// OGCG:   br label %C
// OGCG: C:                                                ; preds = %B, %indirectgoto
// OGCG:   br label %D
// OGCG: D:                                                ; preds = %C, %indirectgoto
// OGCG:   ret void
// OGCG: indirectgoto:                                     ; No predecessors!
// OGCG:   indirectbr ptr poison, [label %D, label %C, label %B, label %A]
