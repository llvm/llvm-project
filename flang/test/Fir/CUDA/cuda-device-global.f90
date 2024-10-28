
// RUN: fir-opt --split-input-file --cuf-device-global %s | FileCheck %s


// -----// IR Dump After CUFLaunchToGPU (cuf-fir-launch-to-gpu) //----- //
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module} {
  fir.global @_QMmtestsEn(dense<[3, 4, 5, 6, 7]> : tensor<5xi32>) {data_attr = #cuf.cuda<device>} : !fir.array<5xi32>

  gpu.module @cuda_device_mod [#nvvm.target] {
  }
}

// CHECK: gpu.module @cuda_device_mod [#nvvm.target] 
// CHECK-NEXT: fir.global @_QMmtestsEn(dense<[3, 4, 5, 6, 7]> : tensor<5xi32>) {data_attr = #cuf.cuda<device>} : !fir.array<5xi32>