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

// CIR:  cir.func {{.*}} @label
// CIR-NEXT:    cir.return

// LLVM:define dso_local void @label
// LLVM-NEXT:  ret void

// OGCG: define dso_local void @label
// OGCG:   br label %labelA
// OGCG: labelA:
// OGCG:   ret void

void multiple_labels() {
labelB:
labelC:
  return;
}

// CIR:  cir.func {{.*}} @multiple_labels
// CIR-NEXT:    cir.return

// LLVM: define dso_local void @multiple_labels(){{.*}}
// LLVM-NEXT:   ret void

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

// CIR:  cir.func {{.*}} @label_in_if
// CIR:      cir.if {{.*}} {
// CIR-NOT:    cir.label
// CIR:        [[LOAD:%.*]] = cir.load align(4) [[COND:%.*]] : !cir.ptr<!s32i>, !s32i
// CIR:        [[INC:%.*]] = cir.inc nsw %3 : !s32i
// CIR:        cir.store align(4) [[INC]], [[COND]] : !s32i, !cir.ptr<!s32i>
// CIR:      }
// CIR:    cir.return

// LLVM: define dso_local void @label_in_if{{.*}}
// LLVM:   br label %3
// LLVM: 3:
// LLVM:   [[LOAD:%.*]] = load i32, ptr [[COND:%.*]], align 4
// LLVM:   [[CMP:%.*]] = icmp ne i32 [[LOAD]], 0
// LLVM:   br i1 [[CMP]], label %6, label %9
// LLVM: 6:
// LLVM:   [[LOAD2:%.*]] = load i32, ptr [[COND]], align 4
// LLVM:   [[ADD1:%.*]] = add nsw i32 [[LOAD2]], 1
// LLVM:   store i32 [[ADD1]], ptr [[COND]], align 4
// LLVM:   br label %9
// LLVM: 9:
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

// CIR:  cir.func {{.*}} @after_return
// CIR:    cir.return
// CIR:  ^bb1:  // no predecessors
// CIR-NEXT:    cir.return

// LLVM: define dso_local void @after_return{{.*}}
// LLVM:   ret void
// LLVM: 1:
// LLVM:   ret void

// OGCG: define dso_local void @after_return
// OGCG:   br label %label
// OGCG: label:
// OGCG:   ret void


void after_unreachable() {
  __builtin_unreachable();
  label:
}

// CIR:  cir.func {{.*}} @after_unreachable
// CIR:    cir.unreachable
// CIR:  ^bb1:
// CIR-NEXT:    cir.return

// LLVM: define dso_local void @after_unreachable{{.*}}
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
// CIR:  cir.func {{.*}} @labelWithoutMatch
// CIR-NEXT:    cir.return
// CIR-NEXT:  }

// LLVM: define dso_local void @labelWithoutMatch
// LLVM-NEXT:   ret void

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

// CIR: cir.func {{.*}} @foo
// CIR:   cir.scope {
// CIR:     %0 = cir.alloca "agg.tmp0" {{.*}} : !cir.ptr<!rec_S>
// CIR-NOT:  cir.label
// CIR:      cir.call @bar()

// LLVM: define dso_local void @foo(){{.*}} {
// LLVM:  %{{.*}} = alloca %struct.S, align 1
// LLVM:  br label %2
// LLVM:2:
// LLVM:  call void @get()
// LLVM:  call void @bar()

// OGCG: define dso_local void @foo()
// OGCG:   %agg.tmp = alloca %struct.S, align 1
// OGCG:   %undef.agg.tmp = alloca %struct.S, align 1
// OGCG:   br label %label
// OGCG: label:
