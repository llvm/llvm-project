// RUN: fir-opt --split-input-file --cuf-transform-device-func %s | FileCheck %s

func.func @_QPsub_device1() attributes {cuf.proc_attr = #cuf.cuda_proc<device>} {
  return
}

func.func @_QPsub_device2(%arg0: !fir.ref<f32> {fir.bindc_name = "i", cuf.proc_attr = #cuf.cuda_proc<device>}) attributes {cuf.proc_attr = #cuf.cuda_proc<device>} {
  %0 = fir.declare %arg0 {uniq_name = "_QFsub1Ei"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %cst = arith.constant 2.000000e+00 : f32
  fir.store %cst to %0 : !fir.ref<f32>
  return
}

func.func @_QPsub_global1() attributes {cuf.proc_attr = #cuf.cuda_proc<global>} {
  %cst = arith.constant 2.000000e+00 : f32
  return
}

func.func @_QPsub_host_device1() attributes {cuf.proc_attr = #cuf.cuda_proc<host_device>} {
  return
}

func.func private @_QMmod1Psub1(!fir.ref<!fir.array<10xi32>> {cuf.data_attr = #cuf.cuda<device>}) attributes {cuf.proc_attr = #cuf.cuda_proc<global>}

// CHECK-LABEL:  module attributes {gpu.container_module}

// CHECK-NOT: func.func @_QPsub_device1()

// CHECK-NOT: func.func @_QPsub_device2(%arg0: !fir.ref<f32> {fir.bindc_name = "i", cuf.data_attr = #cuf.cuda<device>}) attributes {cuf.proc_attr = #cuf.cuda_proc<device>}

// CHECK: func.func @_QPsub_host_device1()

// CHECK-LABEL: gpu.module @cuda_device_mod

// CHECK: gpu.func @_QPsub_device1()

// CHECK: gpu.func @_QPsub_device2(%[[ARG0:.*]]: !fir.ref<f32>) {
// CHECK:   %[[DECL:.*]] = fir.declare %[[ARG0]] {uniq_name = "_QFsub1Ei"} : (!fir.ref<f32>) -> !fir.ref<f32>
// CHECK:   %[[CST:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:   fir.store %[[CST]] to %[[DECL]] : !fir.ref<f32>
// CHECK:   gpu.return
// CHECK: }

// CHECK: gpu.func @_QPsub_global1() kernel

// CHECK: gpu.func @_QPsub_host_device1()

// CHECK: func.func nested @_QMmod1Psub1(!fir.ref<!fir.array<10xi32>> {cuf.data_attr = #cuf.cuda<device>}) attributes {gpu.kernel}

// CHECK: func.func @_QPsub_global1() attributes {cuf.proc_attr = #cuf.cuda_proc<global>}
// CHECK-NEXT: return

// -----

func.func @_QPdevsub() -> i32 attributes {cuf.proc_attr = #cuf.cuda_proc<device>} {
  %c1_i32 = arith.constant 1 : i32
  return %c1_i32 : i32
}

// CHECK-LABEL: gpu.module @cuda_device_mod

// CHECK: gpu.func @_QPdevsub() -> i32

// CHECK: gpu.return %{{.*}} : i32

// -----

func.func @hostFuncUsedInDevice() {
  return
}

func.func @_QPsub_device4() attributes {cuf.proc_attr = #cuf.cuda_proc<device>} {
  fir.call @hostFuncUsedInDevice() : () -> ()
  return
}

// CHECK-LABEL: module attributes {gpu.container_module}
// CHECK: func.func @hostFuncUsedInDevice()
// CHECK: gpu.module @cuda_device_mod
// CHECK: func.func @hostFuncUsedInDevice()
// CHECK: gpu.func @_QPsub_device4()
// CHECK: fir.call @hostFuncUsedInDevice() : () -> ()

// -----

func.func @_QPsub_grid_global1() attributes {cuf.proc_attr = #cuf.cuda_proc<grid_global>} {
  %cst = arith.constant 2.000000e+00 : f32
  return
}

// CHEC-LABEL: gpu.module @cuda_device_mod {
// CHECK: gpu.func @_QPsub_grid_global1() kernel

// CHECK-LABEL: func.func @_QPsub_grid_global1()

// -----

func.func @hostFuncUsedInDevice() {
  return
}

func.func @_QPsub_host() {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  cuf.kernel<<<%c1_i32, %c1_i32>>> (%arg0 : index) = (%c1 : index) to (%c1 : index)  step (%c1 : index) {
    fir.call @hostFuncUsedInDevice() : () -> ()
    "fir.end"() : () -> ()
  }
  return
}

// CHECK-LABEL: func.func @hostFuncUsedInDevice()
// CHECK-LABEL: gpu.module @cuda_device_mod
// CHECK: func.func @hostFuncUsedInDevice()

// -----

func.func @_QPpartialsumshflshflr8(%arg0: !fir.ref<!fir.array<?xf64>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "a"}, %arg1: i32 {fir.bindc_name = "n"}) attributes {cuf.proc_attr = #cuf.cuda_proc<global>} {
  %c2_i32 = arith.constant 2 : i32
  %c10_i32 = arith.constant 10 : i32
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32
  fir.store %arg1 to %1 : !fir.ref<i32>
  %2 = fir.declare %1 dummy_scope %0 {fortran_attrs = #fir.var_attrs<intent_in, value>, uniq_name = "_QFpartialsumshflshflr8En"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %9 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFpartialsumshflshflr8Ei"}
  %10 = fir.declare %9 {uniq_name = "_QFpartialsumshflshflr8Ei"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %13 = fir.alloca i32 {bindc_name = "__builtin_warpsize", uniq_name = "_QM__fortran_builtinsEC__builtin_warpsize"}
  %14 = fir.declare %13 {uniq_name = "_QM__fortran_builtinsEC__builtin_warpsize"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %15 = fir.load %2 : !fir.ref<i32>
  %16 = fir.convert %15 : (i32) -> index
  %17 = arith.cmpi sgt, %16, %c0 : index
  %18 = arith.select %17, %16, %c0 : index
  %19 = fir.shape %18 : (index) -> !fir.shape<1>
  %20 = fir.declare %arg0(%19) dummy_scope %0 {data_attr = #cuf.cuda<device>, uniq_name = "_QFpartialsumshflshflr8Ea"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xf64>>
  cf.br ^bb1
^bb1:  // 2 preds: ^bb0, ^bb2
  %21 = fir.load %10 : !fir.ref<i32>
  %22 = arith.cmpi slt, %21, %c10_i32 : i32
  cf.cond_br %22, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %23 = fir.load %10 : !fir.ref<i32>
  %24 = arith.muli %23, %c2_i32 : i32
  %25 = fir.convert %24 : (i32) -> f64
  %26 = fir.convert %23 : (i32) -> i64
  %27 = fir.array_coor %20(%19) %26 : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>, i64) -> !fir.ref<f64>
  fir.store %25 to %27 : !fir.ref<f64>
  cf.br ^bb1
^bb3:  // pred: ^bb1
  return
}

// CHECK-LABEL: gpu.module @cuda_device_mod
// CHECK: gpu.func @_QPpartialsumshflshflr8(%arg0: !fir.ref<!fir.array<?xf64>>, %arg1: i32) kernel
      
// CHECK: func.func @_QPpartialsumshflshflr8

// -----

func.func @_QPsub_maxtnid() attributes {cuf.launch_bounds = #cuf.launch_bounds<maxTPB = 256 : i64, minBPM = 2 : i64, upperBoundClusterSize = 3 : i64>, cuf.proc_attr = #cuf.cuda_proc<global>} {
  %cst = arith.constant 2.000000e+00 : f32
  return
}

// CHECK: gpu.func @_QPsub_maxtnid() kernel attributes {nvvm.maxntid = array<i32: 256, 1, 1>, nvvm.minctasm = 2 : i64}
