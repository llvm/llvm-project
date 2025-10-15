// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

void label() {
labelA:
  return;
}

// CIR:  cir.func no_proto dso_local @label
// CIR:     cir.br ^bb1
// CIR:  ^bb1:
// CIR:    cir.label "labelA"
// CIR:    cir.return

// LLVM:define dso_local void @label
// LLVM:   br label %1
// LLVM: 1:
// LLVM:  ret void

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
// CIR:    cir.br ^bb1
// CIR:  ^bb1:
// CIR:    cir.label "labelB"
// CIR:    cir.br ^bb2
// CIR:  ^bb2:
// CIR:    cir.label "labelC"
// CIR:    cir.return

// LLVM: define dso_local void @multiple_labels()
// LLVM:   br label %1
// LLVM: 1:
// LLVM:   br label %2
// LLVM: 2:
// LLVM:   ret void

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
// CIR:        cir.br ^bb1
// CIR:      ^bb1:
// CIR:        cir.label "labelD"
// CIR:        [[LOAD:%.*]] = cir.load align(4) [[COND:%.*]] : !cir.ptr<!s32i>, !s32i
// CIR:        [[INC:%.*]] = cir.unary(inc, %3) nsw : !s32i, !s32i
// CIR:        cir.store align(4) [[INC]], [[COND]] : !s32i, !cir.ptr<!s32i>
// CIR:      }
// CIR:    cir.return

// LLVM: define dso_local void @label_in_if
// LLVM:   br label %3
// LLVM: 3:
// LLVM:   [[LOAD:%.*]] = load i32, ptr [[COND:%.*]], align 4
// LLVM:   [[CMP:%.*]] = icmp ne i32 [[LOAD]], 0
// LLVM:   br i1 [[CMP]], label %6, label %10
// LLVM: 6:
// LLVM:   br label %7
// LLVM: 7:
// LLVM:   [[LOAD2:%.*]] = load i32, ptr [[COND]], align 4
// LLVM:   [[ADD1:%.*]] = add nsw i32 [[LOAD2]], 1
// LLVM:   store i32 [[ADD1]], ptr [[COND]], align 4
// LLVM:   br label %10
// LLVM: 10:
// LLVM:   br label %11
// LLVM: 11:
// LLVM:  ret void

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

// LLVM: define dso_local void @after_return
// LLVM:   br label %1
// LLVM: 1:
// LLVM:   ret void
// LLVM: 2:
// LLVM:   br label %1

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

// LLVM: define dso_local void @after_unreachable
// LLVM:   unreachable
// LLVM: 1:
// LLVM:   ret void

// OGCG: define dso_local void @after_unreachable
// OGCG:   unreachable
// OGCG: label:
// OGCG:   ret void

void labelWithoutMatch() {
end:
  return;
}
// CIR:  cir.func no_proto dso_local @labelWithoutMatch
// CIR:    cir.br ^bb1
// CIR:  ^bb1:
// CIR:    cir.label "end"
// CIR:    cir.return
// CIR:  }

// LLVM: define dso_local void @labelWithoutMatch
// LLVM:   br label %1
// LLVM: 1:
// LLVM:   ret void

// OGCG: define dso_local void @labelWithoutMatch
// OGCG:   br label %end
// OGCG: end:
// OGCG:   ret void

struct S {};
struct S get();
void bar(struct S);

void foo() {
  {
    label:
      bar(get());
  }
}

// CIR: cir.func no_proto dso_local @foo
// CIR:   cir.scope {
// CIR:     %0 = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["agg.tmp0"]
// CIR:      cir.br ^bb1
// CIR:    ^bb1:
// CIR:     cir.label "label"

// LLVM:define dso_local void @foo() {
// LLVM:  [[ALLOC:%.*]] = alloca %struct.S, i64 1, align 1
// LLVM:  br label %2
// LLVM:2:
// LLVM:  br label %3
// LLVM:3:
// LLVM:  [[CALL:%.*]] = call %struct.S @get()
// LLVM:  store %struct.S [[CALL]], ptr [[ALLOC]], align 1
// LLVM:  [[LOAD:%.*]] = load %struct.S, ptr [[ALLOC]], align 1
// LLVM:  call void @bar(%struct.S [[LOAD]])

// OGCG: define dso_local void @foo()
// OGCG:   %agg.tmp = alloca %struct.S, align 1
// OGCG:   %undef.agg.tmp = alloca %struct.S, align 1
// OGCG:   br label %label
// OGCG: label:
