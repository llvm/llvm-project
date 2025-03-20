// RUN: fir-opt --split-input-file --target-rewrite="target=x86_64-unknown-linux-gnu" %s | FileCheck %s

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {

gpu.module @testmod {
  gpu.func @_QPvcpowdk(%arg0: !fir.ref<complex<f64>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "a"}) attributes {cuf.proc_attr = #cuf.cuda_proc<global>} {
    %0 = fir.alloca i64
    %1 = fir.load %0 : !fir.ref<i64>
    %2 = fir.load %arg0 : !fir.ref<complex<f64>>
    %3 = fir.call @_FortranAzpowk(%2, %1) fastmath<contract> : (complex<f64>, i64) -> complex<f64>
    gpu.return
  }
  func.func private @_FortranAzpowk(complex<f64>, i64) -> complex<f64> attributes {fir.bindc_name = "_FortranAzpowk", fir.runtime}
}

// CHECK-LABEL: gpu.func @_QPvcpowdk
// CHECK: %{{.*}} = fir.call @_FortranAzpowk(%{{.*}}, %{{.*}}, %{{.*}}) : (f64, f64, i64) -> tuple<f64, f64>
// CHECK: func.func private @_FortranAzpowk(f64, f64, i64) -> tuple<f64, f64> attributes {fir.bindc_name = "_FortranAzpowk", fir.runtime}
}

// -----

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {

gpu.module @testmod {
  gpu.func @_QPtest(%arg0: complex<f64>) -> (complex<f64>) {
    gpu.return %arg0 : complex<f64>
  }
}
}

// CHECK-LABEL: gpu.func @_QPtest
// CHECK-SAME: (%arg0: f64, %arg1: f64) -> tuple<f64, f64> {
// CHECK: gpu.return %{{.*}} : tuple<f64, f64>

// -----
module attributes {gpu.container_module, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {

gpu.module @testmod {
  gpu.func @_QPtest(%arg0: complex<f64>) -> () kernel {
    gpu.return
  }
}

func.func @main(%arg0: complex<f64>) {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = llvm.mlir.constant(0 : i32) : i32
  gpu.launch_func  @testmod::@_QPtest blocks in (%0, %0, %0) threads in (%0, %0, %0) : i64 dynamic_shared_memory_size %1 args(%arg0 : complex<f64>)
  return
}

}

// CHECK-LABEL: gpu.func @_QPtest
// CHECK-SAME: (%arg0: f64, %arg1: f64) kernel {
// CHECK: gpu.return
// CHECK: gpu.launch_func  @testmod::@_QPtest blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}}) : i64 dynamic_shared_memory_size %{{.*}} args(%{{.*}} : f64, %{{.*}} : f64)
