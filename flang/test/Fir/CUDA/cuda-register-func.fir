// RUN: fir-opt %s | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @cuda_device_mod {
    gpu.func @_QPsub_device1() kernel {
      gpu.return
    }
    gpu.func @_QPsub_device2(%arg0: !fir.ref<f32>) kernel {
      gpu.return
    }
  }
  llvm.func internal @__cudaFortranConstructor() {
    cuf.register_kernel @cuda_device_mod::@_QPsub_device1
    cuf.register_kernel @cuda_device_mod::@_QPsub_device2
    llvm.return
  }
}

// CHECK: cuf.register_kernel @cuda_device_mod::@_QPsub_device1
// CHECK: cuf.register_kernel @cuda_device_mod::@_QPsub_device2
