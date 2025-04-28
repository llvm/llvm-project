// Tests converting `fir.do_concurrent` ops to `fir.do_loop` ops.

// RUN: fir-opt --split-input-file --simplify-fir-operations %s | FileCheck %s

func.func @dc_1d(%i_lb: index, %i_ub: index, %i_st: index) {
  fir.do_concurrent {
    %i = fir.alloca i32
    fir.do_concurrent.loop (%i_iv) = (%i_lb) to (%i_ub) step (%i_st) {
      %0 = fir.convert %i_iv : (index) -> i32
      fir.store %0 to %i : !fir.ref<i32>
    }
  }
  return
}

// CHECK-LABEL:   func.func @dc_1d(
// CHECK-SAME:                     %[[I_LB:[^[:space:]]+]]: index,
// CHECK-SAME:                     %[[I_UB:[^[:space:]]+]]: index,
// CHECK-SAME:                     %[[I_ST:[^[:space:]]+]]: index) {

// CHECK:           %[[I:.*]] = fir.alloca i32

// CHECK:           fir.do_loop %[[I_IV:.*]] = %[[I_LB]] to %[[I_UB]] step %[[I_ST]] unordered {
// CHECK:             %[[I_IV_CVT:.*]] = fir.convert %[[I_IV]] : (index) -> i32
// CHECK:             fir.store %[[I_IV_CVT]] to %[[I]] : !fir.ref<i32>
// CHECK:           }

// CHECK:           return
// CHECK:         }

// -----

func.func @dc_2d(%i_lb: index, %i_ub: index, %i_st: index,
                 %j_lb: index, %j_ub: index, %j_st: index) {
  llvm.br ^bb1

^bb1:
  fir.do_concurrent {
    %i = fir.alloca i32
    %j = fir.alloca i32
    fir.do_concurrent.loop
      (%i_iv, %j_iv) = (%i_lb, %j_lb) to (%i_ub, %j_ub) step (%i_st, %j_st) {
      %0 = fir.convert %i_iv : (index) -> i32
      fir.store %0 to %i : !fir.ref<i32>

      %1 = fir.convert %j_iv : (index) -> i32
      fir.store %1 to %j : !fir.ref<i32>
    }
  }
  return
}

// CHECK-LABEL:   func.func @dc_2d(
// CHECK-SAME:                     %[[I_LB:[^[:space:]]+]]: index,
// CHECK-SAME:                     %[[I_UB:[^[:space:]]+]]: index,
// CHECK-SAME:                     %[[I_ST:[^[:space:]]+]]: index,
// CHECK-SAME:                     %[[J_LB:[^[:space:]]+]]: index,
// CHECK-SAME:                     %[[J_UB:[^[:space:]]+]]: index,
// CHECK-SAME:                     %[[J_ST:[^[:space:]]+]]: index) {

// CHECK:           %[[I:.*]] = fir.alloca i32
// CHECK:           %[[J:.*]] = fir.alloca i32
// CHECK:           llvm.br ^bb1

// CHECK:         ^bb1:
// CHECK:           fir.do_loop %[[I_IV:.*]] = %[[I_LB]] to %[[I_UB]] step %[[I_ST]] unordered {
// CHECK:             fir.do_loop %[[J_IV:.*]] = %[[J_LB]] to %[[J_UB]] step %[[J_ST]] unordered {
// CHECK:               %[[I_IV_CVT:.*]] = fir.convert %[[I_IV]] : (index) -> i32
// CHECK:               fir.store %[[I_IV_CVT]] to %[[I]] : !fir.ref<i32>
// CHECK:               %[[J_IV_CVT:.*]] = fir.convert %[[J_IV]] : (index) -> i32
// CHECK:               fir.store %[[J_IV_CVT]] to %[[J]] : !fir.ref<i32>
// CHECK:             }
// CHECK:           }

// CHECK:           return
// CHECK:         }

// -----

func.func @dc_2d_reduction(%i_lb: index, %i_ub: index, %i_st: index,
                 %j_lb: index, %j_ub: index, %j_st: index) {
  %sum = fir.alloca i32

  fir.do_concurrent {
    %i = fir.alloca i32
    %j = fir.alloca i32
    fir.do_concurrent.loop
      (%i_iv, %j_iv) = (%i_lb, %j_lb) to (%i_ub, %j_ub) step (%i_st, %j_st) 
      reduce(#fir.reduce_attr<add> -> %sum : !fir.ref<i32>) {
      %0 = fir.convert %i_iv : (index) -> i32
      fir.store %0 to %i : !fir.ref<i32>

      %1 = fir.convert %j_iv : (index) -> i32
      fir.store %1 to %j : !fir.ref<i32>
    }
  }
  return
}

// CHECK-LABEL:   func.func @dc_2d_reduction(
// CHECK-SAME:                     %[[I_LB:[^[:space:]]+]]: index,
// CHECK-SAME:                     %[[I_UB:[^[:space:]]+]]: index,
// CHECK-SAME:                     %[[I_ST:[^[:space:]]+]]: index,
// CHECK-SAME:                     %[[J_LB:[^[:space:]]+]]: index,
// CHECK-SAME:                     %[[J_UB:[^[:space:]]+]]: index,
// CHECK-SAME:                     %[[J_ST:[^[:space:]]+]]: index) {

// CHECK:           %[[I:.*]] = fir.alloca i32
// CHECK:           %[[J:.*]] = fir.alloca i32
// CHECK:           %[[SUM:.*]] = fir.alloca i32

// CHECK:           fir.do_loop %[[I_IV:.*]] = %[[I_LB]] to %[[I_UB]] step %[[I_ST]] unordered reduce({{.*}}<add>] -> %[[SUM]] : !fir.ref<i32>) {
// CHECK:             fir.do_loop %[[J_IV:.*]] = %[[J_LB]] to %[[J_UB]] step %[[J_ST]] unordered reduce({{.*}}<add>] -> %[[SUM]] : !fir.ref<i32>) {
// CHECK:               %[[I_IV_CVT:.*]] = fir.convert %[[I_IV]] : (index) -> i32
// CHECK:               fir.store %[[I_IV_CVT]] to %[[I]] : !fir.ref<i32>
// CHECK:               %[[J_IV_CVT:.*]] = fir.convert %[[J_IV]] : (index) -> i32
// CHECK:               fir.store %[[J_IV_CVT]] to %[[J]] : !fir.ref<i32>
// CHECK:               fir.result
// CHECK:             }
// CHECK:             fir.result
// CHECK:           }
// CHECK:           return
// CHECK:         }
