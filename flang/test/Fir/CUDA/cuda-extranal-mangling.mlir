// RUN: fir-opt --split-input-file --external-name-interop %s | FileCheck %s

gpu.module @cuda_device_mod {
  gpu.func @_QPfoo() {
    fir.call @_QPthreadfence() fastmath<contract> : () -> ()
    gpu.return
  }
  func.func private @_QPthreadfence() attributes {cuf.proc_attr = #cuf.cuda_proc<device>}
}

// CHECK-LABEL: gpu.func @_QPfoo
// CHECK: fir.call @threadfence_()
// CHECK: func.func private @threadfence_()
