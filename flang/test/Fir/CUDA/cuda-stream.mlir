// RUN: fir-opt --split-input-file %s | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @cuda_device_mod {
    gpu.func @_QMmod1Psub1() kernel {
      gpu.return
    }
  }
  func.func @_QMmod1Phost_sub() {
    %0 = fir.alloca i64
    %1 = arith.constant 1 : index
    %asyncTok = cuf.stream_cast %0 : !fir.ref<i64>
    gpu.launch_func [%asyncTok] @cuda_device_mod::@_QMmod1Psub1 blocks in (%1, %1, %1) threads in (%1, %1, %1) args() {cuf.proc_attr = #cuf.cuda_proc<grid_global>}
    return
  }
}

// CHECK-LABEL: func.func @_QMmod1Phost_sub()
// CHECK: %[[STREAM:.*]] = fir.alloca i64
// CHECK: %[[TOKEN:.*]] = cuf.stream_cast %[[STREAM]] : <i64>
// CHECK: gpu.launch_func [%[[TOKEN]]] @cuda_device_mod::@_QMmod1Psub1
