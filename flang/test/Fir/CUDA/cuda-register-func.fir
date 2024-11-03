// RUN: fir-opt --cuf-add-constructor %s | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @cuda_device_mod {
    gpu.func @_QPsub_device1() kernel {
      gpu.return
    }
    gpu.func @_QPsub_device2(%arg0: !fir.ref<f32>) kernel {
      gpu.return
    }
  }
}

// CHECK-LABEL: llvm.func internal @__cudaFortranConstructor()
// CHECK: %[[MOD_HANDLE:.*]] = cuf.register_module @cuda_device_mod -> !llvm.ptr
// CHECK: cuf.register_kernel @cuda_device_mod::@_QPsub_device1(%[[MOD_HANDLE]] : !llvm.ptr)
// CHECK: cuf.register_kernel @cuda_device_mod::@_QPsub_device2(%[[MOD_HANDLE]] : !llvm.ptr)
