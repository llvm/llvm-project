// RUN: fir-opt --split-input-file --cuf-function-rewrite %s | FileCheck %s

gpu.module @cuda_device_mod {
  func.func @_QMmtestsPdo2(%arg0: !fir.ref<i32> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "c"}, %arg1: !fir.ref<i32> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "i"}) attributes {cuf.proc_attr = #cuf.cuda_proc<host_device>} {
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = fir.dummy_scope : !fir.dscope
    %5 = fir.declare %arg0 dummy_scope %0 {uniq_name = "_QMmtestsFdo2Ec"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
    %8 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QMmtestsFdo2Ei"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
    %13 = fir.call @on_device() proc_attrs<bind_c> fastmath<contract> : () -> !fir.logical<4>
    %14 = fir.convert %13 : (!fir.logical<4>) -> i1
    fir.if %14 {
      fir.store %c1_i32 to %5 : !fir.ref<i32>
    } else {
      fir.store %c2_i32 to %5 : !fir.ref<i32>
    }
    return
  }
}

// CHECK-LABEL: gpu.module @cuda_device_mod
// CHECK: func.func @_QMmtestsPdo2
// CHECK: fir.if %true

// -----

func.func @_QMmtestsPdo3(%arg0: !fir.ref<i32> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "c"}, %arg1: !fir.ref<i32> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "i"}) attributes {cuf.proc_attr = #cuf.cuda_proc<host_device>} {
  %c2_i32 = arith.constant 2 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %5 = fir.declare %arg0 dummy_scope %0 {uniq_name = "_QMmtestsFdo2Ec"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %8 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QMmtestsFdo2Ei"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %13 = fir.call @on_device() proc_attrs<bind_c> fastmath<contract> : () -> !fir.logical<4>
  %14 = fir.convert %13 : (!fir.logical<4>) -> i1
  fir.if %14 {
    fir.store %c1_i32 to %5 : !fir.ref<i32>
  } else {
    fir.store %c2_i32 to %5 : !fir.ref<i32>
  }
  return
}

// CHECK-LABEL: func.func @_QMmtestsPdo3
// CHECK: fir.if %false
