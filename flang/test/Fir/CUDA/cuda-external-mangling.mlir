// RUN: fir-opt --split-input-file --external-name-interop %s | FileCheck %s

module @mod attributes {gpu.container_module} {

gpu.module @cuda_device_mod {
  gpu.func @_QPfoo() kernel {
    fir.call @_QPthreadfence() fastmath<contract> : () -> ()
    gpu.return
  }
  func.func private @_QPthreadfence() attributes {cuf.proc_attr = #cuf.cuda_proc<device>}
}

func.func @test() -> () {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = llvm.mlir.constant(0 : i32) : i32
  gpu.launch_func  @cuda_device_mod::@_QPfoo blocks in (%0, %0, %0) threads in (%0, %0, %0) : i64 dynamic_shared_memory_size %1
  return
}

// CHECK-LABEL: gpu.func @foo_()
// CHECK: fir.call @threadfence_()
// CHECK: func.func private @threadfence_()
// CHECK: gpu.launch_func  @cuda_device_mod::@foo_ 

}
