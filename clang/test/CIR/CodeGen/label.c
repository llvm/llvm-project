// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

void label() {
labelA:
  return;
}

// CIR:  cir.func no_proto dso_local @label
// CIR:    cir.label "labelA"
// CIR:    cir.return

// Note: We are not lowering to LLVM IR via CIR at this stage because that
// process depends on the GotoSolver.

// OGCG: define dso_local void @label
// OGCG:   br label %labelA
// OGCG: labelA:
// OGCG:   ret void

void multiple_labels() {
labelB:
labelC:
  return;
}

// CIR:  cir.func no_proto dso_local @multiple_labels
// CIR:    cir.label "labelB"
// CIR:    cir.br ^bb1
// CIR:  ^bb1:  // pred: ^bb0
// CIR:    cir.label "labelC"
// CIR:    cir.return

// OGCG: define dso_local void @multiple_labels
// OGCG:   br label %labelB
// OGCG: labelB:
// OGCG:   br label %labelC
// OGCG: labelC:
// OGCG:   ret void

void label_in_if(int cond) {
  if (cond) {
labelD:
    cond++;
  }
}

// CIR:  cir.func dso_local @label_in_if
// CIR:      cir.if {{.*}} {
// CIR:        cir.label "labelD"
// CIR:        [[LOAD:%.*]] = cir.load align(4) [[COND:%.*]] : !cir.ptr<!s32i>, !s32i
// CIR:        [[INC:%.*]] = cir.unary(inc, %3) nsw : !s32i, !s32i
// CIR:        cir.store align(4) [[INC]], [[COND]] : !s32i, !cir.ptr<!s32i>
// CIR:      }
// CIR:    cir.return

// OGCG: define dso_local void @label_in_if
// OGCG: if.then:
// OGCG:   br label %labelD
// OGCG: labelD:
// OGCG:   [[LOAD:%.*]] = load i32, ptr [[COND:%.*]], align 4
// OGCG:   [[INC:%.*]] = add nsw i32 %1, 1
// OGCG:   store i32 [[INC]], ptr [[COND]], align 4
// OGCG:   br label %if.end
// OGCG: if.end:
// OGCG:   ret void

void after_return() {
  return;
  label:
}

// CIR:  cir.func no_proto dso_local @after_return
// CIR:    cir.br ^bb1
// CIR:  ^bb1:  // 2 preds: ^bb0, ^bb2
// CIR:    cir.return
// CIR:  ^bb2:  // no predecessors
// CIR:    cir.label "label"
// CIR:    cir.br ^bb1

// OGCG: define dso_local void @after_return
// OGCG:   br label %label
// OGCG: label:
// OGCG:   ret void


void after_unreachable() {
  __builtin_unreachable();
  label:
}

// CIR:  cir.func no_proto dso_local @after_unreachable
// CIR:    cir.unreachable
// CIR:  ^bb1:
// CIR:    cir.label "label"
// CIR:    cir.return

// OGCG: define dso_local void @after_unreachable
// OGCG:   unreachable
// OGCG: label:
// OGCG:   ret void
