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

// -----

fir.local {type = local} @local_localizer : i32

fir.local {type = local_init} @local_init_localizer : i32 copy {
^bb0(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>):
    %0 = fir.load %arg0 : !fir.ref<i32>
    fir.store %0 to %arg1 : !fir.ref<i32>
    fir.yield(%arg1 : !fir.ref<i32>)
}

func.func @do_concurrent_locality_specs() {
  %3 = fir.alloca i32 {bindc_name = "local_init_var", uniq_name = "_QFdo_concurrentElocal_init_var"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFdo_concurrentElocal_init_var"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5 = fir.alloca i32 {bindc_name = "local_var", uniq_name = "_QFdo_concurrentElocal_var"}
  %6:2 = hlfir.declare %5 {uniq_name = "_QFdo_concurrentElocal_var"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 1 : index
  fir.do_concurrent {
    %9 = fir.alloca i32 {bindc_name = "i"}
    %10:2 = hlfir.declare %9 {uniq_name = "_QFdo_concurrentEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    fir.do_concurrent.loop (%arg0) = (%c1) to (%c10) step (%c1) local(@local_localizer %6#0 -> %arg1, @local_init_localizer %4#0 -> %arg2 : !fir.ref<i32>, !fir.ref<i32>) {
      %11 = fir.convert %arg0 : (index) -> i32
      fir.store %11 to %10#0 : !fir.ref<i32>
      %13:2 = hlfir.declare %arg1 {uniq_name = "_QFdo_concurrentElocal_var"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %15:2 = hlfir.declare %arg2 {uniq_name = "_QFdo_concurrentElocal_init_var"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %17 = fir.load %10#0 : !fir.ref<i32>
      %c5_i32 = arith.constant 5 : i32
      %18 = arith.cmpi slt, %17, %c5_i32 : i32
      fir.if %18 {
        %c42_i32 = arith.constant 42 : i32
        hlfir.assign %c42_i32 to %13#0 : i32, !fir.ref<i32>
      } else {
        %c84_i32 = arith.constant 84 : i32
        hlfir.assign %c84_i32 to %15#0 : i32, !fir.ref<i32>
      }
    }
  }
  return
}

// CHECK-LABEL:   func.func @do_concurrent_locality_specs() {
// CHECK:           %[[LOC_INIT_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "{{.*}}Elocal_init_var"}
// CHECK:           fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
// Verify localization of the `local` var.
// CHECK:             %[[PRIV_LOC_ALLOC:.*]] = fir.alloca i32

// Verify localization of the `local_init` var.
// CHECK:             %[[PRIV_LOC_INIT_ALLOC:.*]] = fir.alloca i32
// CHECK:             %[[LOC_INIT_VAL:.*]] = fir.load %[[LOC_INIT_DECL]]#0 : !fir.ref<i32>
// CHECK:             fir.store %[[LOC_INIT_VAL]] to %[[PRIV_LOC_INIT_ALLOC]] : !fir.ref<i32>

// CHECK:             %[[VAL_15:.*]]:2 = hlfir.declare %[[PRIV_LOC_ALLOC]]
// CHECK:             %[[VAL_16:.*]]:2 = hlfir.declare %[[PRIV_LOC_INIT_ALLOC]]

// CHECK:               hlfir.assign %{{.*}} to %[[VAL_15]]#0 : i32, !fir.ref<i32>
// CHECK:               hlfir.assign %{{.*}} to %[[VAL_16]]#0 : i32, !fir.ref<i32>
// CHECK:           }
// CHECK:           return
// CHECK:         }
