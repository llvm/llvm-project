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
      reduce(@add_reduction_i32 #fir.reduce_attr<add> %sum -> %sum_arg : !fir.ref<i32>) {
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
// CHECK-SAME:        (%[[I_IV:.*]], %[[J_IV:.*]]) = (%[[I_LB]], %[[J_LB]]) to (%[[I_UB]], %[[J_UB]]) step (%[[I_ST]], %[[J_ST]]) reduce(@add_reduction_i32 #fir.reduce_attr<add> %[[SUM]] -> %{{.*}} : !fir.ref<i32>) {
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

func.func @do_concurrent_with_locality_specs() {
  %3 = fir.alloca i32 {bindc_name = "local_init_var"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFdo_concurrentElocal_init_var"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5 = fir.alloca i32 {bindc_name = "local_var"}
  %6:2 = hlfir.declare %5 {uniq_name = "_QFdo_concurrentElocal_var"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  fir.do_concurrent {
    %9 = fir.alloca i32 {bindc_name = "i"}
    %10:2 = hlfir.declare %9 {uniq_name = "_QFdo_concurrentEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

    fir.do_concurrent.loop (%arg0) = (%c1) to (%c10) step (%c1)
      local(@local_privatizer %6#0 -> %arg1, @local_init_privatizer %4#0 -> %arg2 : !fir.ref<i32>, !fir.ref<i32>) {
      %11 = fir.convert %arg0 : (index) -> i32
      fir.store %11 to %10#0 : !fir.ref<i32>
      %13:2 = hlfir.declare %arg1 {uniq_name = "_QFdo_concurrentElocal_var"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %15:2 = hlfir.declare %arg2 {uniq_name = "_QFdo_concurrentElocal_init_var"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    }
  }
  return
}

// CHECK-LABEL:   func.func @do_concurrent_with_locality_specs() {
// CHECK:           %[[LOC_INIT_ALLOC:.*]] = fir.alloca i32 {bindc_name = "local_init_var"}
// CHECK:           %[[LOC_INIT_DECL:.*]]:2 = hlfir.declare %[[LOC_INIT_ALLOC]]

// CHECK:           %[[LOC_ALLOC:.*]] = fir.alloca i32 {bindc_name = "local_var"}
// CHECK:           %[[LOC_DECL:.*]]:2 = hlfir.declare %[[LOC_ALLOC]]

// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C10:.*]] = arith.constant 10 : index

// CHECK:           fir.do_concurrent {
// CHECK:             %[[DC_I_ALLOC:.*]] = fir.alloca i32 {bindc_name = "i"}
// CHECK:             %[[DC_I_DECL:.*]]:2 = hlfir.declare %[[DC_I_ALLOC]]

// CHECK:             fir.do_concurrent.loop (%[[IV:.*]]) = (%[[C1]]) to 
// CHECK-SAME:          (%[[C10]]) step (%[[C1]])
// CHECK-SAME:          local(@[[LOCAL_PRIV_SYM]] %[[LOC_DECL]]#0 -> %[[LOC_ARG:[^,]*]],
// CHECK-SAME:                @[[LOCAL_INIT_PRIV_SYM]] %[[LOC_INIT_DECL]]#0 -> %[[LOC_INIT_ARG:.*]] :
// CHECK-SAME:          !fir.ref<i32>, !fir.ref<i32>) {

// CHECK:               %[[IV_CVT:.*]] = fir.convert %[[IV]] : (index) -> i32
// CHECK:               fir.store %[[IV_CVT]] to %[[DC_I_DECL]]#0 : !fir.ref<i32>

// CHECK:               %[[LOC_PRIV_DECL:.*]]:2 = hlfir.declare %[[LOC_ARG]]
// CHECK:               %[[LOC_INIT_PRIV_DECL:.*]]:2 = hlfir.declare %[[LOC_INIT_ARG]]
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @dc_reduce() {
  %3 = fir.alloca i32 {bindc_name = "s", uniq_name = "dc_reduce"}
  %4:2 = hlfir.declare %3 {uniq_name = "dc_reduce"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %c1 = arith.constant 1 : index 
  fir.do_concurrent {
    fir.do_concurrent.loop (%arg0) = (%c1) to (%c1) step (%c1) reduce(byref @add_reduction_i32 #fir.reduce_attr<add> %4#0 -> %arg1 : !fir.ref<i32>) {
    }
  }
  return
}

// CHECK-LABEL:  func.func @dc_reduce() {
// CHECK:          %[[S_ALLOC:.*]] = fir.alloca i32 {bindc_name = "s", uniq_name = "dc_reduce"}
// CHECK:          %[[S_DECL:.*]]:2 = hlfir.declare %[[S_ALLOC]] {uniq_name = "dc_reduce"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:          fir.do_concurrent {
// CHECK:            fir.do_concurrent.loop (%{{.*}}) = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) reduce(byref @add_reduction_i32 #fir.reduce_attr<add> %[[S_DECL]]#0 -> %[[S_ARG:.*]] : !fir.ref<i32>) {
// CHECK:            }
// CHECK:          }
// CHECK:          return
// CHECK:        }

func.func @dc_reduce_2() {
  %3 = fir.alloca i32 {bindc_name = "s", uniq_name = "dc_reduce"}
  %4:2 = hlfir.declare %3 {uniq_name = "dc_reduce"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

  %5 = fir.alloca i32 {bindc_name = "m", uniq_name = "dc_reduce"}
  %6:2 = hlfir.declare %5 {uniq_name = "dc_reduce"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

  %c1 = arith.constant 1 : index 

  fir.do_concurrent {
    fir.do_concurrent.loop (%arg0) = (%c1) to (%c1) step (%c1) 
      reduce(@add_reduction_i32 #fir.reduce_attr<add> %4#0 -> %arg1,
             @mul_reduction_i32 #fir.reduce_attr<multiply> %6#0 -> %arg2
             : !fir.ref<i32>, !fir.ref<i32>) {
    }
  }

  return
}

// CHECK-LABEL:  func.func @dc_reduce_2() {
// CHECK:          %[[S_ALLOC:.*]] = fir.alloca i32 {bindc_name = "s", uniq_name = "dc_reduce"}
// CHECK:          %[[S_DECL:.*]]:2 = hlfir.declare %[[S_ALLOC]] {uniq_name = "dc_reduce"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

// CHECK:          %[[M_ALLOC:.*]] = fir.alloca i32 {bindc_name = "m", uniq_name = "dc_reduce"}
// CHECK:          %[[M_DECL:.*]]:2 = hlfir.declare %[[M_ALLOC]] {uniq_name = "dc_reduce"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:          fir.do_concurrent {
// CHECK:            fir.do_concurrent.loop (%{{.*}}) = (%{{.*}}) to (%{{.*}}) step (%{{[^[:space:]]+}}) 
// CHECK-SAME:         reduce(
// CHECK-SAME:           @add_reduction_i32 #fir.reduce_attr<add> %[[S_DECL]]#0 -> %[[S_ARG:[^,]+]],
// CHECK-SAME:           @mul_reduction_i32 #fir.reduce_attr<multiply> %[[M_DECL]]#0 -> %[[M_ARG:[^[:space:]]+]]
// CHECK-SAME:           : !fir.ref<i32>, !fir.ref<i32>) {
// CHECK:            }
// CHECK:          }
// CHECK:          return
// CHECK:        }

