// Test fir.do_concurrent operation parse, verify (no errors), and unparse

// RUN: fir-opt %s | fir-opt | FileCheck %s

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

// CHECK-LABEL: func.func @dc_1d
// CHECK-SAME:    (%[[I_LB:.*]]: index, %[[I_UB:.*]]: index, %[[I_ST:.*]]: index)
// CHECK:         fir.do_concurrent {
// CHECK:           %[[I:.*]] = fir.alloca i32
// CHECK:           fir.do_concurrent.loop (%[[I_IV:.*]]) = (%[[I_LB]]) to (%[[I_UB]]) step (%[[I_ST]]) {
// CHECK:             %[[I_IV_CVT:.*]] = fir.convert %[[I_IV]] : (index) -> i32
// CHECK:             fir.store %[[I_IV_CVT]] to %[[I]] : !fir.ref<i32>
// CHECK:           }
// CHECK:         }

func.func @dc_2d(%i_lb: index, %i_ub: index, %i_st: index,
                 %j_lb: index, %j_ub: index, %j_st: index) {
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

// CHECK-LABEL: func.func @dc_2d
// CHECK-SAME:    (%[[I_LB:.*]]: index, %[[I_UB:.*]]: index, %[[I_ST:.*]]: index, %[[J_LB:.*]]: index, %[[J_UB:.*]]: index, %[[J_ST:.*]]: index)
// CHECK:         fir.do_concurrent {
// CHECK:           %[[I:.*]] = fir.alloca i32
// CHECK:           %[[J:.*]] = fir.alloca i32
// CHECK:           fir.do_concurrent.loop
// CHECK-SAME:        (%[[I_IV:.*]], %[[J_IV:.*]]) = (%[[I_LB]], %[[J_LB]]) to (%[[I_UB]], %[[J_UB]]) step (%[[I_ST]], %[[J_ST]]) {
// CHECK:             %[[I_IV_CVT:.*]] = fir.convert %[[I_IV]] : (index) -> i32
// CHECK:             fir.store %[[I_IV_CVT]] to %[[I]] : !fir.ref<i32>
// CHECK:             %[[J_IV_CVT:.*]] = fir.convert %[[J_IV]] : (index) -> i32
// CHECK:             fir.store %[[J_IV_CVT]] to %[[J]] : !fir.ref<i32>
// CHECK:           }
// CHECK:         }

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

// CHECK-LABEL: func.func @dc_2d_reduction
// CHECK-SAME:    (%[[I_LB:.*]]: index, %[[I_UB:.*]]: index, %[[I_ST:.*]]: index, %[[J_LB:.*]]: index, %[[J_UB:.*]]: index, %[[J_ST:.*]]: index)

// CHECK:         %[[SUM:.*]] = fir.alloca i32

// CHECK:         fir.do_concurrent {
// CHECK:           %[[I:.*]] = fir.alloca i32
// CHECK:           %[[J:.*]] = fir.alloca i32
// CHECK:           fir.do_concurrent.loop
// CHECK-SAME:        (%[[I_IV:.*]], %[[J_IV:.*]]) = (%[[I_LB]], %[[J_LB]]) to (%[[I_UB]], %[[J_UB]]) step (%[[I_ST]], %[[J_ST]]) reduce(#fir.reduce_attr<add> -> %[[SUM]] : !fir.ref<i32>) {
// CHECK:             %[[I_IV_CVT:.*]] = fir.convert %[[I_IV]] : (index) -> i32
// CHECK:             fir.store %[[I_IV_CVT]] to %[[I]] : !fir.ref<i32>
// CHECK:             %[[J_IV_CVT:.*]] = fir.convert %[[J_IV]] : (index) -> i32
// CHECK:             fir.store %[[J_IV_CVT]] to %[[J]] : !fir.ref<i32>
// CHECK:           }
// CHECK:         }


fir.local {type = local} @local_privatizer : i32

// CHECK:   fir.local {type = local} @[[LOCAL_PRIV_SYM:local_privatizer]] : i32

fir.local {type = local_init} @local_init_privatizer : i32 copy {
^bb0(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>):
    %0 = fir.load %arg0 : !fir.ref<i32>
    fir.store %0 to %arg1 : !fir.ref<i32>
    fir.yield(%arg1 : !fir.ref<i32>)
}

// CHECK:   fir.local {type = local_init} @[[LOCAL_INIT_PRIV_SYM:local_init_privatizer]] : i32
// CHECK:   ^bb0(%[[ORIG_VAL:.*]]: !fir.ref<i32>, %[[LOCAL_VAL:.*]]: !fir.ref<i32>):
// CHECK:      %[[ORIG_VAL_LD:.*]] = fir.load %[[ORIG_VAL]]
// CHECK:      fir.store %[[ORIG_VAL_LD]] to %[[LOCAL_VAL]] : !fir.ref<i32>
// CHECK:      fir.yield(%[[LOCAL_VAL]] : !fir.ref<i32>)
// CHECK:   }
