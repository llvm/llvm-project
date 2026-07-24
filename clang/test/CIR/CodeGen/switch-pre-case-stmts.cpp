// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

void pre_case_label() {
  switch (1) {
  LABEL1:;
  case 1:;
  }
}

// CIR-LABEL: cir.func{{.*}} @_Z14pre_case_labelv
// CIR: cir.switch({{.*}} : !s32i) {
// CIR:    cir.br ^bb1
// CIR:  ^bb1:
// CIR:    cir.label "LABEL1"
// CIR:    cir.case(equal, [#cir.int<1> : !s32i]) {
// CIR:      cir.yield loc(#loc7)
// CIR:    }
// CIR:    cir.yield
// CIR:  }

// LLVM-LABEL: define{{.*}} void @_Z14pre_case_labelv
// LLVM: switch i32 1, label %[[END:.*]] [
// LLVM:   i32 1, label %[[CASE1:.*]]
// LLVM: ]
// LLVM: [[LABEL:.*]]:{{.*}}; No predecessors!
// LLVM: br label %[[JUMPS_TO_CASE1:.*]]
// LLVM: [[JUMPS_TO_CASE1]]:
// LLVM:   br label %[[CASE1]]
// CIR inserts a bunch of 'empty' blocks all over the place that just jump to
// the next, so unifying these check blocks/checking more is pretty
// awful/doesn't result in good checks.
// LLVM: [[END]]:
// LLVM:   br label

// OGCG-LABEL: define{{.*}} void @_Z14pre_case_labelv
// OGCG: switch i32 1, label %[[END:.*]] [
// OGCG:    i32 1, label %[[CASE1:.*]]
// OGCG:  ]
// OGCG: [[LABEL:.*]]:{{.*}}; No predecessors!
// OGCG:   br label %[[CASE1]]
// OGCG: [[CASE1]]:
// OGCG:   br label %[[END]]
// OGCG: [[END]]:
// OGCG:   ret void

void multiple_pre_case_labels() {
  switch (1) {
  LABEL1:;
  LABEL2:;
  case 1:;
  }
}

// CIR-LABEL: cir.func{{.*}} @_Z24multiple_pre_case_labelsv
// CIR:  cir.switch({{.*}} : !s32i) {
// CIR:    cir.br ^bb1
// CIR:  ^bb1:
// CIR:    cir.label "LABEL1"
// CIR:    cir.br ^bb2
// CIR:  ^bb2:
// CIR:    cir.label "LABEL2"
// CIR:    cir.case(equal, [#cir.int<1> : !s32i]) {
// CIR:      cir.yield
// CIR:    }
// CIR:    cir.yield
// CIR:  }

// LLVM-LABEL: define{{.*}} @_Z24multiple_pre_case_labelsv
// LLVM:   switch i32 1, label %[[END:.*]] [
// LLVM:     i32 1, label %[[CASE1:.*]]
// LLVM:   ]
// LLVM: [[LABEL1:.*]]:
// LLVM:   br label %[[LABEL2:.*]]
// LLVM: [[LABEL2]]:
// LLVM:   br label %[[JUMPS_TO_CASE1:.*]]
// LLVM: [[JUMPS_TO_CASE1]]:
// LLVM:   br label %[[CASE1]]
// LLVM: [[CASE1]]:
// LLVM:   br label %[[JUMPS_TO_END:.*]]
// LLVM: [[JUMPS_TO_END]]:
// LLVM:   br label %[[END]]
// LLVM: [[END]]:
// LLVM:   br label

// OGCG-LABEL: define{{.*}} @_Z24multiple_pre_case_labelsv
// OGCG:   switch i32 1, label %[[END:.*]] [
// OGCG:     i32 1, label %[[CASE1:.*]]
// OGCG:   ]
// OGCG: [[LABEL1:.*]]:{{.*}}; No predecessors!
// OGCG:   br label %[[LABEL2:.*]]
// OGCG: [[LABEL2]]:
// OGCG:   br label %[[CASE1]]
// OGCG: [[CASE1]]:
// OGCG:   br label %[[END]]
// OGCG: [[END]]:
// OGCG:   ret void
// OGCG: }

void pre_case_goto() {
  switch (1) {
    goto end;
    case 1:;
    end:;
  }
}

// CIR-LABEL: cir.func{{.*}} @_Z13pre_case_gotov
// CIR: cir.switch({{.*}} : !s32i) {
// CIR:   cir.goto "end"
// CIR: ^bb1:
// CIR:   cir.case(equal, [#cir.int<1> : !s32i]) {
// CIR:     cir.br ^bb1
// CIR:   ^bb1:
// CIR:     cir.label "end"
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.yield
// CIR: }

// Classic-codegen manages to remove the switch, but we don't have that sort of
// analysis working right.  So this variant has a switch still in place.
// LLVM-LABEL: define{{.*}} @_Z13pre_case_gotov
// LLVM:  switch i32 1, label %[[END:.*]] [
// LLVM:    i32 1, label %[[CASE1:.*]]
// LLVM:  ]
//
// OGCG-LABEL: define{{.*}} @_Z13pre_case_gotov
// OGCG:  br label %[[END:.*]]
// OGCG: [[END]]:
// OGCG:  ret void

void pre_case_if(int cond) {
  switch (1) {
    if (cond) {}
    case 1:;
  }
}

// CIR-LABEL: cir.func{{.*}} @_Z11pre_case_ifi
// CIR: cir.switch({{.*}} : !s32i) {
// CIR:   cir.scope {
// CIR:     cir.load
// CIR:     cir.cast int_to_bool
// CIR:     cir.if {{.*}} {
// CIR:     }
// CIR:   }
// CIR:   cir.case(equal, [#cir.int<1> : !s32i]) {
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.yield
// CIR: }

// Once again, classic-codegen manages to make this a 'noop' and remove it.
// LLVM-LABEL: define{{.*}} @_Z11pre_case_ifi
// LLVM:  switch i32 1, label %[[END:.*]] [
// LLVM:    i32 1, label %[[CASE1:.*]]
// LLVM:  ]
// LLVM: [[ENTRY:.*]]:{{.*}}; No predecessors!
// LLVM: br label %[[IF_BLOCK:.*]]
// LLVM: [[IF_BLOCK]]:
// LLVM: load i32
// LLVM: icmp ne i32
// LLVM: br i1 %{{.*}}, label %[[TRUE:.*]], label %[[FALSE:.*]]
// LLVM: [[TRUE]]:
// LLVM: br label %[[FALSE]]
// LLVM: [[FALSE]]:
// LLVM: br label %[[JUMPS_TO_CASE1:.*]]
// LLVM: [[JUMPS_TO_CASE1:.*]]:
// LLVM: br label %[[CASE1]]
// LLVM: [[CASE1]]:
// LLVM: [[END]]:

// OGCG-LABEL: define{{.*}} @_Z11pre_case_ifi
// OGCG: alloca i32
// OGCG: store i32 %{{.*}}, ptr %
// OGCG: ret void

void pre_case_return() {
  switch (1) {
    return;
    case 1:;
  }
}

// CIR-LABEL: cir.func{{.*}} @_Z15pre_case_returnv
// CIR: cir.switch({{.*}} : !s32i) {
// CIR:   cir.return
// CIR: ^bb1:
// CIR:   cir.case(equal, [#cir.int<1> : !s32i]) {
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.yield
// CIR: }

// Once again, classic codegen skips this entirely, but CIR doesn't.
// LLVM-LABEL: define{{.*}} @_Z15pre_case_returnv
// LLVM: switch i32 1, label %[[END:.*]] [
// LLVM:   i32 1, label %[[CASE1:.*]]
// LLVM: ]
// LLVM: [[ENTRY:.*]]:{{.*}}; No predecessors!
// LLVM:   ret void
// LLVM: [[CASE1]]:
// LLVM: [[END]]:

// OGCG-LABEL: define{{.*}} @_Z15pre_case_returnv
// OGCG: ret void

void pre_case_break() {
  switch (1) {
    break;
    case 1:;
  }
}

// CIR-LABEL: cir.func{{.*}} @_Z14pre_case_breakv
// CIR: cir.switch({{.*}} : !s32i) {
// CIR:   cir.break
// CIR: ^bb1:
// CIR:   cir.case(equal, [#cir.int<1> : !s32i]) {
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.yield
// CIR: }

// LLVM-LABEL: define{{.*}} @_Z14pre_case_breakv
// LLVM: switch i32 1, label %[[END:.*]] [
// LLVM:   i32 1, label %[[CASE1:.*]]
// LLVM: ]
// LLVM: [[ENTRY:.*]]:{{.*}}; No predecessors!
// LLVM: br label %[[END]]
// LLVM: [[CASE1]]:
// LLVM: [[END]]:
//
// OGCG-LABEL: define{{.*}} @_Z14pre_case_breakv
// OGCG: ret void

void label_only_switch() {
  switch (1) {
  LABEL:;
  }
}

// CIR-LABEL: cir.func{{.*}} @_Z17label_only_switchv
// CIR:  cir.switch({{.*}} : !s32i) {
// CIR:    cir.br ^bb1
// CIR:  ^bb1:
// CIR:    cir.label "LABEL"
// CIR:    cir.yield
// CIR:  }
// CIR:}

// LLVM-LABEL: define{{.*}} @_Z17label_only_switchv
// LLVM:   switch i32 1, label %[[END:.*]] [
// LLVM:   ]
// LLVM: [[LABEL:.*]]:{{.*}}; No predecessors!
// LLVM:   br label %[[JUMPS_TO_END:.*]]
// LLVM: [[JUMPS_TO_END]]:
// LLVM:   br label %[[END]]

// OGCG-LABEL: define{{.*}} @_Z17label_only_switchv
// OGCG:   switch i32 1, label %[[END:.*]] [
// OGCG:   ]
// OGCG: [[LABEL:.*]]:{{.*}}; No predecessors!
// OGCG:   br label %[[END]]
// OGCG: [[END]]:
// OGCG:   ret void

void external_goto_into_pre_case(int cond) {
  if (cond) goto LABEL;
  switch (1) {
  LABEL:;
  case 1:;
  }
}
// CIR-LABEL: cir.func{{.*}} @_Z27external_goto_into_pre_casei
// CIR: cir.if {{.*}} {
// CIR:   cir.goto "LABEL"
// CIR: cir.switch({{.*}} : !s32i) {
// CIR:   cir.br ^bb1
// CIR: ^bb1:
// CIR:   cir.label "LABEL"
// CIR:   cir.case(equal, [#cir.int<1> : !s32i]) {
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.yield
// CIR: }

// LLVM-LABEL: define{{.*}} @_Z27external_goto_into_pre_casei
// LLVM: br i1 %{{.*}}, label %[[TRUE:.*]], label %[[FALSE:.*]]
// LLVM: [[TRUE]]:
// LLVM:   br label %[[LABEL:.*]]
// LLVM: [[FALSE]]:
// Some empty blocks that just do jumps removed here.
// LLVM: switch i32 1, label %[[END:.*]] [
// LLVM:   i32 1, label %[[CASE1:.*]]
// LLVM: ]
// LLVM: [[ENTRY:.*]]:{{.*}}; No predecessors!
// LLVM:   br label %[[LABEL]]
// LLVM: [[LABEL]]:
// LLVM: br label %[[CASE1]]
// LLVM: [[CASE1]]:
// LLVM: [[END]]:

// OGCG-LABEL: define{{.*}} @_Z27external_goto_into_pre_casei
// OGCG: br i1 %{{.*}}, label %[[TRUE:.*]], label %[[FALSE:.*]]
// OGCG: [[TRUE]]:
// OGCG:   br label %[[LABEL:.*]]
// OGCG: [[FALSE]]:
// OGCG: switch i32 1, label %[[END:.*]] [
// OGCG:   i32 1, label %[[CASE1:.*]]
// OGCG: ]
// OGCG: [[LABEL]]:
// OGCG:   br label %[[CASE1]]
// OGCG: [[CASE1]]:
// OGCG:   br label %[[END]]
// OGCG: [[END]]:
// OGCG:   ret void

void external_goto_into_pre_case_empty(int cond) {
  if (cond) goto LABEL;
  switch (1) {
  LABEL:;
  }
}
// CIR-LABEL: cir.func{{.*}} @_Z33external_goto_into_pre_case_emptyi
// CIR: cir.if {{.*}} {
// CIR:   cir.goto "LABEL"
// CIR: cir.switch({{.*}} : !s32i) {
// CIR:   cir.br ^bb1
// CIR: ^bb1:
// CIR:   cir.label "LABEL"
// CIR:   cir.yield
// CIR: }

// LLVM-LABEL: define{{.*}} @_Z33external_goto_into_pre_case_emptyi
// LLVM: br i1 %{{.*}}, label %[[TRUE:.*]], label %[[FALSE:.*]]
// LLVM: [[TRUE]]:
// LLVM:   br label %[[LABEL:.*]]
// LLVM: [[FALSE]]:
// Some empty blocks that just do jumps removed here.
// LLVM: switch i32 1, label %[[END:.*]] [
// LLVM: ]
// LLVM: [[ENTRY:.*]]:{{.*}}; No predecessors!
// LLVM:   br label %[[LABEL]]
// LLVM: [[LABEL]]:
// LLVM: br label %[[END]]
// LLVM: [[END]]:

// OGCG-LABEL: define{{.*}} @_Z33external_goto_into_pre_case_emptyi
// OGCG: br i1 %{{.*}}, label %[[TRUE:.*]], label %[[FALSE:.*]]
// OGCG: [[TRUE]]:
// OGCG:   br label %[[LABEL:.*]]
// OGCG: [[FALSE]]:
// OGCG: switch i32 1, label %[[END:.*]] [
// OGCG: ]
// OGCG: [[LABEL]]:
// OGCG:   br label %[[END]]
// OGCG: [[END]]:
// OGCG:   ret void
