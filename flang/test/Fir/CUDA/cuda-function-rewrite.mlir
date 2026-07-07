// RUN: fir-opt --split-input-file --cuf-function-rewrite %s | FileCheck %s

// Test the bind(c) name "on_device" in device context.
gpu.module @cuda_device_mod {
  func.func private @on_device() -> !fir.logical<4>
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

// Test the bind(c) name "on_device" on host side.
func.func private @on_device() -> !fir.logical<4>
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

// -----

// Test on_device() with Fortran name mangling (_QPon_device) in device context.
gpu.module @acc_device_mod {
  func.func private @_QPon_device() -> !fir.logical<4>
  func.func @_QMmtestPsub_device() {
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = fir.alloca i32
    %13 = fir.call @_QPon_device() fastmath<contract> : () -> !fir.logical<4>
    %14 = fir.convert %13 : (!fir.logical<4>) -> i1
    fir.if %14 {
      fir.store %c1_i32 to %0 : !fir.ref<i32>
    } else {
      fir.store %c2_i32 to %0 : !fir.ref<i32>
    }
    return
  }
}

// CHECK-LABEL: gpu.module @acc_device_mod
// CHECK: func.func @_QMmtestPsub_device
// CHECK: fir.if %true

// -----

// Test _QPon_device on host side.
func.func private @_QPon_device() -> !fir.logical<4>
func.func @_QMmtestPsub_host() {
  %c2_i32 = arith.constant 2 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.alloca i32
  %13 = fir.call @_QPon_device() fastmath<contract> : () -> !fir.logical<4>
  %14 = fir.convert %13 : (!fir.logical<4>) -> i1
  fir.if %14 {
    fir.store %c1_i32 to %0 : !fir.ref<i32>
  } else {
    fir.store %c2_i32 to %0 : !fir.ref<i32>
  }
  return
}

// CHECK-LABEL: func.func @_QMmtestPsub_host
// CHECK: fir.if %false

// -----

// Test externally-mangled on_device_ (after ExternalNameConversion) in device
// context. The original name is recovered from the fir.internal_name attribute.
gpu.module @acc_extname_device_mod {
  func.func private @on_device_() -> !fir.logical<4> attributes {fir.internal_name = "_QPon_device"}
  func.func @_QMmtestPsub_extname_device() {
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = fir.alloca i32
    %13 = fir.call @on_device_() fastmath<contract> : () -> !fir.logical<4>
    %14 = fir.convert %13 : (!fir.logical<4>) -> i1
    fir.if %14 {
      fir.store %c1_i32 to %0 : !fir.ref<i32>
    } else {
      fir.store %c2_i32 to %0 : !fir.ref<i32>
    }
    return
  }
}

// CHECK-LABEL: gpu.module @acc_extname_device_mod
// CHECK: func.func @_QMmtestPsub_extname_device
// CHECK: fir.if %true

// -----

// Test on_device_ on host side (original name recovered from fir.internal_name).
func.func private @on_device_() -> !fir.logical<4> attributes {fir.internal_name = "_QPon_device"}
func.func @_QMmtestPsub_extname_host() {
  %c2_i32 = arith.constant 2 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.alloca i32
  %13 = fir.call @on_device_() fastmath<contract> : () -> !fir.logical<4>
  %14 = fir.convert %13 : (!fir.logical<4>) -> i1
  fir.if %14 {
    fir.store %c1_i32 to %0 : !fir.ref<i32>
  } else {
    fir.store %c2_i32 to %0 : !fir.ref<i32>
  }
  return
}

// CHECK-LABEL: func.func @_QMmtestPsub_extname_host
// CHECK: fir.if %false

// -----

// A user-defined procedure named on_device (with a body) must not be folded.
func.func @_QPon_device() -> !fir.logical<4> {
  %true = arith.constant true
  %0 = fir.convert %true : (i1) -> !fir.logical<4>
  return %0 : !fir.logical<4>
}
func.func @_QMmtestPsub_userdef() -> !fir.logical<4> {
  %13 = fir.call @_QPon_device() fastmath<contract> : () -> !fir.logical<4>
  return %13 : !fir.logical<4>
}

// CHECK-LABEL: func.func @_QMmtestPsub_userdef
// CHECK: fir.call @_QPon_device()

// -----

// A call whose signature does not match the intrinsic (extra argument) must not
// be folded.
func.func private @_QPon_device(i32) -> !fir.logical<4>
func.func @_QMmtestPsub_badsig(%arg0: i32) -> !fir.logical<4> {
  %13 = fir.call @_QPon_device(%arg0) fastmath<contract> : (i32) -> !fir.logical<4>
  return %13 : !fir.logical<4>
}

// CHECK-LABEL: func.func @_QMmtestPsub_badsig
// CHECK: fir.call @_QPon_device(%arg0)
