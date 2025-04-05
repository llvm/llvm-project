// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

void l0() {
  for (;;) {
  }
}

// CIR: cir.func @l0
// CIR:   cir.scope {
// CIR:     cir.for : cond {
// CIR:       %[[TRUE:.*]] = cir.const #true
// CIR:       cir.condition(%[[TRUE]])
// CIR:     } body {
// CIR:       cir.yield
// CIR:     } step {
// CIR:       cir.yield
// CIR:     }
// CIR:   }
// CIR:   cir.return
// CIR: }

// LLVM: define void @l0()
// LLVM:   br label %[[LABEL1:.*]]
// LLVM: [[LABEL1]]:
// LLVM:   br label %[[LABEL2:.*]]
// LLVM: [[LABEL2]]:
// LLVM:   br i1 true, label %[[LABEL3:.*]], label %[[LABEL5:.*]]
// LLVM: [[LABEL3]]:
// LLVM:   br label %[[LABEL4:.*]]
// LLVM: [[LABEL4]]:
// LLVM:   br label %[[LABEL2]]
// LLVM: [[LABEL5]]:
// LLVM:   br label %[[LABEL6:.*]]
// LLVM: [[LABEL6]]:
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z2l0v()
// OGCG: entry:
// OGCG:   br label %[[FOR_COND:.*]]
// OGCG: [[FOR_COND]]:
// OGCG:   br label %[[FOR_COND]]

void l1() {
  for (int i = 0; ; ) {
  }
}

// CIR:      cir.func @l1
// CIR-NEXT:   cir.scope {
// CIR-NEXT:     %[[I:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
// CIR-NEXT:     %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT:     cir.store %[[ZERO]], %[[I]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:     cir.for : cond {
// CIR-NEXT:       %[[TRUE:.*]] = cir.const #true
// CIR-NEXT:       cir.condition(%[[TRUE]])
// CIR-NEXT:     } body {
// CIR-NEXT:       cir.yield
// CIR-NEXT:     } step {
// CIR-NEXT:       cir.yield
// CIR-NEXT:     }
// CIR-NEXT:   }
// CIR-NEXT:   cir.return
// CIR-NEXT: }

// LLVM: define void @l1()
// LLVM:   %[[I:.*]] = alloca i32, i64 1, align 4
// LLVM:   br label %[[LABEL1:.*]]
// LLVM: [[LABEL1]]:
// LLVM:   store i32 0, ptr %[[I]], align 4
// LLVM:   br label %[[LABEL2:.*]]
// LLVM: [[LABEL2]]:
// LLVM:   br i1 true, label %[[LABEL3:.*]], label %[[LABEL5:.*]]
// LLVM: [[LABEL3]]:
// LLVM:   br label %[[LABEL4:.*]]
// LLVM: [[LABEL4]]:
// LLVM:   br label %[[LABEL2]]
// LLVM: [[LABEL5]]:
// LLVM:   br label %[[LABEL6:.*]]
// LLVM: [[LABEL6]]:
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z2l1v()
// OGCG: entry:
// OGCG:   %[[I:.*]] = alloca i32, align 4
// OGCG:   store i32 0, ptr %[[I]], align 4
// OGCG:   br label %[[FOR_COND:.*]]
// OGCG: [[FOR_COND]]:
// OGCG:   br label %[[FOR_COND]]

void l2() {
  for (;;) {
    int i = 0;
  }
}

// CIR:      cir.func @l2
// CIR-NEXT:   cir.scope {
// CIR-NEXT:     cir.for : cond {
// CIR-NEXT:       %[[TRUE:.*]] = cir.const #true
// CIR-NEXT:       cir.condition(%[[TRUE]])
// CIR-NEXT:     } body {
// CIR-NEXT:       cir.scope {
// CIR-NEXT:         %[[I:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
// CIR-NEXT:         %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT:         cir.store %[[ZERO]], %[[I]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:       }
// CIR-NEXT:       cir.yield
// CIR-NEXT:     } step {
// CIR-NEXT:       cir.yield
// CIR-NEXT:     }
// CIR-NEXT:   }
// CIR-NEXT:   cir.return
// CIR-NEXT: }

// LLVM: define void @l2()
// LLVM:   %[[I:.*]] = alloca i32, i64 1, align 4
// LLVM:   br label %[[LABEL1:.*]]
// LLVM: [[LABEL1]]:
// LLVM:   br label %[[LABEL2:.*]]
// LLVM: [[LABEL2]]:
// LLVM:   br i1 true, label %[[LABEL3:.*]], label %[[LABEL5:.*]]
// LLVM: [[LABEL3]]:
// LLVM:   store i32 0, ptr %[[I]], align 4
// LLVM:   br label %[[LABEL4:.*]]
// LLVM: [[LABEL4]]:
// LLVM:   br label %[[LABEL2]]
// LLVM: [[LABEL5]]:
// LLVM:   br label %[[LABEL6:.*]]
// LLVM: [[LABEL6]]:
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z2l2v()
// OGCG: entry:
// OGCG:   %[[I:.*]] = alloca i32, align 4
// OGCG:   br label %[[FOR_COND:.*]]
// OGCG: [[FOR_COND]]:
// OGCG:   store i32 0, ptr %[[I]], align 4
// OGCG:   br label %[[FOR_COND]]

// This is the same as l2 but without a compound statement for the body.
void l3() {
  for (;;)
    int i = 0;
}

// CIR:      cir.func @l3
// CIR-NEXT:   cir.scope {
// CIR-NEXT:     %[[I:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
// CIR-NEXT:     cir.for : cond {
// CIR-NEXT:       %[[TRUE:.*]] = cir.const #true
// CIR-NEXT:       cir.condition(%[[TRUE]])
// CIR-NEXT:     } body {
// CIR-NEXT:       %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT:       cir.store %[[ZERO]], %[[I]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:       cir.yield
// CIR-NEXT:     } step {
// CIR-NEXT:       cir.yield
// CIR-NEXT:     }
// CIR-NEXT:   }
// CIR-NEXT:   cir.return
// CIR-NEXT: }

// LLVM: define void @l3()
// LLVM:   %[[I:.*]] = alloca i32, i64 1, align 4
// LLVM:   br label %[[LABEL1:.*]]
// LLVM: [[LABEL1]]:
// LLVM:   br label %[[LABEL2:.*]]
// LLVM: [[LABEL2]]:
// LLVM:   br i1 true, label %[[LABEL3:.*]], label %[[LABEL5:.*]]
// LLVM: [[LABEL3]]:
// LLVM:   store i32 0, ptr %[[I]], align 4
// LLVM:   br label %[[LABEL4:.*]]
// LLVM: [[LABEL4]]:
// LLVM:   br label %[[LABEL2]]
// LLVM: [[LABEL5]]:
// LLVM:   br label %[[LABEL6:.*]]
// LLVM: [[LABEL6]]:
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z2l3v()
// OGCG: entry:
// OGCG:   %[[I:.*]] = alloca i32, align 4
// OGCG:   br label %[[FOR_COND:.*]]
// OGCG: [[FOR_COND]]:
// OGCG:   store i32 0, ptr %[[I]], align 4
// OGCG:   br label %[[FOR_COND]]

void test_do_while_false() {
  do {
  } while (0);
}

// CIR: cir.func @test_do_while_false()
// CIR-NEXT:   cir.scope {
// CIR-NEXT:     cir.do {
// CIR-NEXT:       cir.yield
// CIR-NEXT:     } while {
// CIR-NEXT:       %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT:       %[[FALSE:.*]] = cir.cast(int_to_bool, %[[ZERO]] : !s32i), !cir.bool
// CIR-NEXT:       cir.condition(%[[FALSE]])

// LLVM: define void @test_do_while_false()
// LLVM:   br label %[[LABEL1:.*]]
// LLVM: [[LABEL1]]:
// LLVM:   br label %[[LABEL3:.*]]
// LLVM: [[LABEL2:.*]]:
// LLVM:   br i1 false, label %[[LABEL3]], label %[[LABEL4:.*]]
// LLVM: [[LABEL3]]:
// LLVM:   br label %[[LABEL2]]
// LLVM: [[LABEL4]]:
// LLVM:   br label %[[LABEL5:.*]]
// LLVM: [[LABEL5]]:
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z19test_do_while_falsev()
// OGCG: entry:
// OGCG:   br label %[[DO_BODY:.*]]
// OGCG: [[DO_BODY]]:
// OGCG:   br label %[[DO_END:.*]]
// OGCG: [[DO_END]]:
// OGCG:   ret void

void test_empty_while_true() {
  while (true) {
    return;
  }
}

// CIR: cir.func @test_empty_while_true()
// CIR-NEXT:   cir.scope {
// CIR-NEXT:     cir.while {
// CIR-NEXT:       %[[TRUE:.*]] = cir.const #true
// CIR-NEXT:       cir.condition(%[[TRUE]])
// CIR-NEXT:     } do {
// CIR-NEXT:       cir.scope {
// CIR-NEXT:         cir.return
// CIR-NEXT:       }
// CIR-NEXT:       cir.yield

// LLVM: define void @test_empty_while_true()
// LLVM:   br label %[[LABEL1:.*]]
// LLVM: [[LABEL1]]:
// LLVM:   br label %[[LABEL2:.*]]
// LLVM: [[LABEL2]]:
// LLVM:   br i1 true, label %[[LABEL3:.*]], label %[[LABEL6:.*]]
// LLVM: [[LABEL3]]:
// LLVM:   br label %[[LABEL4]]
// LLVM: [[LABEL4]]:
// LLVM:   ret void
// LLVM: [[LABEL5:.*]]:
// LLVM-SAME: ; No predecessors!
// LLVM:   br label %[[LABEL2:.*]]
// LLVM: [[LABEL6]]:
// LLVM:   br label %[[LABEL7:.*]]
// LLVM: [[LABEL7]]:
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z21test_empty_while_truev()
// OGCG: entry:
// OGCG:   br label %[[WHILE_BODY:.*]]
// OGCG: [[WHILE_BODY]]:
// OGCG:   ret void

void unreachable_after_continue() {
  for (;;) {
    continue;
    int x = 1;
  }
}

// CIR: cir.func @unreachable_after_continue
// CIR:   cir.scope {
// CIR:     cir.for : cond {
// CIR:       %[[TRUE:.*]] = cir.const #true
// CIR:       cir.condition(%[[TRUE]])
// CIR:     } body {
// CIR:       cir.scope {
// CIR:         %[[X:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init] {alignment = 4 : i64}
// CIR:         cir.continue
// CIR:       ^bb1:  // no predecessors
// CIR:         %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:         cir.store %[[ONE]], %[[X]] : !s32i, !cir.ptr<!s32i>
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:     } step {
// CIR:       cir.yield
// CIR:     }
// CIR:   }
// CIR:   cir.return
// CIR: }

// LLVM: define void @unreachable_after_continue()
// LLVM:   %[[X:.*]] = alloca i32, i64 1, align 4
// LLVM:   br label %[[LABEL1:.*]]
// LLVM: [[LABEL1]]:
// LLVM:   br label %[[LABEL2:.*]]
// LLVM: [[LABEL2]]:
// LLVM:   br i1 true, label %[[LABEL3:.*]], label %[[LABEL8:.*]]
// LLVM: [[LABEL3]]:
// LLVM:   br label %[[LABEL4:.*]]
// LLVM: [[LABEL4]]:
// LLVM:   br label %[[LABEL7:.*]]
// LLVM: [[LABEL5:.*]]:
// LLVM-SAME: ; No predecessors!
// LLVM:   store i32 1, ptr %[[X]], align 4
// LLVM:   br label %[[LABEL6:.*]]
// LLVM: [[LABEL6]]:
// LLVM:   br label %[[LABEL7:.*]]
// LLVM: [[LABEL7]]:
// LLVM:   br label %[[LABEL2]]
// LLVM: [[LABEL8]]:
// LLVM:   br label %[[LABEL9:]]
// LLVM: [[LABEL9]]:
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z26unreachable_after_continuev()
// OGCG: entry:
// OGCG:   %[[X:.*]] = alloca i32, align 4
// OGCG:   br label %[[FOR_COND:.*]]
// OGCG: [[FOR_COND]]:
// OGCG:   br label %[[FOR_COND]]

void unreachable_after_break() {
  for (;;) {
    break;
    int x = 1;
  }
}

// CIR: cir.func @unreachable_after_break
// CIR:   cir.scope {
// CIR:     cir.for : cond {
// CIR:       %[[TRUE:.*]] = cir.const #true
// CIR:       cir.condition(%[[TRUE]])
// CIR:     } body {
// CIR:       cir.scope {
// CIR:         %[[X:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init] {alignment = 4 : i64}
// CIR:         cir.break
// CIR:       ^bb1:  // no predecessors
// CIR:         %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:         cir.store %[[ONE]], %[[X]] : !s32i, !cir.ptr<!s32i>
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:     } step {
// CIR:       cir.yield
// CIR:     }
// CIR:   }
// CIR:   cir.return
// CIR: }

// LLVM: define void @unreachable_after_break()
// LLVM:   %[[X:.*]] = alloca i32, i64 1, align 4
// LLVM:   br label %[[LABEL1:.*]]
// LLVM: [[LABEL1]]:
// LLVM:   br label %[[LABEL2:.*]]
// LLVM: [[LABEL2]]:
// LLVM:   br i1 true, label %[[LABEL3:.*]], label %[[LABEL8:.*]]
// LLVM: [[LABEL3]]:
// LLVM:   br label %[[LABEL4:.*]]
// LLVM: [[LABEL4]]:
// LLVM:   br label %[[LABEL8]]
// LLVM: [[LABEL5:.*]]:
// LLVM-SAME: ; No predecessors!
// LLVM:   store i32 1, ptr %[[X]], align 4
// LLVM:   br label %[[LABEL6:.*]]
// LLVM: [[LABEL6]]:
// LLVM:   br label %[[LABEL7:.*]]
// LLVM: [[LABEL7]]:
// LLVM:   br label %[[LABEL2]]
// LLVM: [[LABEL8]]:
// LLVM:   br label %[[LABEL9:]]
// LLVM: [[LABEL9]]:
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z23unreachable_after_breakv()
// OGCG: entry:
// OGCG:   %[[X:.*]] = alloca i32, align 4
// OGCG:   br label %[[FOR_COND:.*]]
// OGCG: [[FOR_COND]]:
// OGCG:   br label %[[FOR_END:.*]]
// OGCG: [[FOR_END]]:
// OGCG:   ret void
