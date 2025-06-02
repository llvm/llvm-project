// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: cir-opt --mlir-print-ir-before=cir-flatten-cfg --cir-flatten-cfg %t.cir -o %t.flattened.before.cir 2> %t.before
// RUN: FileCheck --input-file=%t.before %s --check-prefix=BEFORE
// RUN: cir-opt --mlir-print-ir-after=cir-flatten-cfg --cir-flatten-cfg %t.cir -o %t.flattened.after.cir 2> %t.after
// RUN: FileCheck --input-file=%t.after %s --check-prefix=AFTER

void swf(int a) {
  switch (int b = 3; a) {
    case 3:
      b = b * 2;
      break;
    case 4 ... 5:
      b = b * 3;
      break;
    default:
      break;
  }

}

// BEFORE:  cir.func @_Z3swfi
// BEFORE:   %[[VAR_B:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init] {alignment = 4 : i64}
// BEFORE:   %[[CONST_3:.*]] = cir.const #cir.int<3> : !s32i
// BEFORE:   cir.switch (%[[COND:.*]] : !s32i) {
// BEFORE:     cir.case(equal, [#cir.int<3> : !s32i]) {
// BEFORE:       %[[LOAD_B_EQ:.*]] = cir.load{{.*}} %[[VAR_B]] : !cir.ptr<!s32i>, !s32i
// BEFORE:       %[[CONST_2:.*]] = cir.const #cir.int<2> : !s32i
// BEFORE:       %[[MUL_EQ:.*]] = cir.binop(mul, %[[LOAD_B_EQ]], %[[CONST_2]]) nsw : !s32i
// BEFORE:       cir.store{{.*}} %[[MUL_EQ]], %[[VAR_B]] : !s32i, !cir.ptr<!s32i>
// BEFORE:       cir.break
// BEFORE:     }
// BEFORE:     cir.case(range, [#cir.int<4> : !s32i, #cir.int<5> : !s32i]) {
// BEFORE:       %[[LOAD_B_RANGE:.*]] = cir.load{{.*}} %[[VAR_B]] : !cir.ptr<!s32i>, !s32i
// BEFORE:       %[[CONST_3_RANGE:.*]] = cir.const #cir.int<3> : !s32i
// BEFORE:       %[[MUL_RANGE:.*]] = cir.binop(mul, %[[LOAD_B_RANGE]], %[[CONST_3_RANGE]]) nsw : !s32i
// BEFORE:       cir.store{{.*}} %[[MUL_RANGE]], %[[VAR_B]] : !s32i, !cir.ptr<!s32i>
// BEFORE:       cir.break
// BEFORE:     }
// BEFORE:     cir.case(default, []) {
// BEFORE:       cir.break
// BEFORE:     }
// BEFORE:     cir.yield
// BEFORE:   }
// BEFORE: }
// BEFORE: cir.return

// AFTER: cir.func @_Z3swfi
// AFTER:  %[[VAR_A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init] {alignment = 4 : i64}
// AFTER:  cir.store{{.*}} %arg0, %[[VAR_A]] : !s32i, !cir.ptr<!s32i>
// AFTER:  %[[VAR_B:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init] {alignment = 4 : i64}
// AFTER:  %[[CONST_3:.*]] = cir.const #cir.int<3> : !s32i
// AFTER:  cir.store{{.*}} %[[CONST_3]], %[[VAR_B]] : !s32i, !cir.ptr<!s32i>
// AFTER:  cir.switch.flat %[[COND:.*]] : !s32i, ^bb[[#BB6:]] [
// AFTER:    3: ^bb[[#BB4:]],
// AFTER:    4: ^bb[[#BB5:]],
// AFTER:    5: ^bb[[#BB5:]]
// AFTER:  ]
// AFTER:  ^bb[[#BB4]]:
// AFTER:  %[[LOAD_B_EQ:.*]] = cir.load{{.*}} %[[VAR_B]] : !cir.ptr<!s32i>, !s32i
// AFTER:  %[[CONST_2:.*]] = cir.const #cir.int<2> : !s32i
// AFTER:  %[[MUL_EQ:.*]] = cir.binop(mul, %[[LOAD_B_EQ]], %[[CONST_2]]) nsw : !s32i
// AFTER:  cir.store{{.*}} %[[MUL_EQ]], %[[VAR_B]] : !s32i, !cir.ptr<!s32i>
// AFTER:  cir.br ^bb[[#BB7:]]
// AFTER:  ^bb[[#BB5]]:
// AFTER:  %[[LOAD_B_RANGE:.*]] = cir.load{{.*}} %[[VAR_B]] : !cir.ptr<!s32i>, !s32i
// AFTER:  %[[CONST_3_AGAIN:.*]] = cir.const #cir.int<3> : !s32i
// AFTER:  %[[MUL_RANGE:.*]] = cir.binop(mul, %[[LOAD_B_RANGE]], %[[CONST_3_AGAIN]]) nsw : !s32i
// AFTER:  cir.store{{.*}} %[[MUL_RANGE]], %[[VAR_B]] : !s32i, !cir.ptr<!s32i>
// AFTER:  cir.br ^bb[[#BB7]]
// AFTER: ^bb[[#BB6]]:
// AFTER: cir.br ^bb[[#BB7]]
// AFTER: ^bb[[#BB7]]:
// AFTER: cir.br ^bb[[#BB8:]]
// AFTER: ^bb[[#BB8]]:
// AFTER: cir.return
// AFTER: }

