
// RUN: fir-opt --split-input-file --cuf-device-global %s | FileCheck %s


module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module} {
  fir.global @_QMmtestsEn(dense<[3, 4, 5, 6, 7]> : tensor<5xi32>) {data_attr = #cuf.cuda<device>} : !fir.array<5xi32>

  gpu.module @cuda_device_mod {
  }
}

// CHECK: gpu.module @cuda_device_mo
// CHECK-NEXT: fir.global @_QMmtestsEn(dense<[3, 4, 5, 6, 7]> : tensor<5xi32>) {data_attr = #cuf.cuda<device>} : !fir.array<5xi32>

// -----

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module} {
  fir.global @_QMm1ECb(dense<[90, 100, 110]> : tensor<3xi32>) constant : !fir.array<3xi32>
  fir.global @_QMm2ECc(dense<[100, 200, 300]> : tensor<3xi32>) constant : !fir.array<3xi32>
}

// CHECK: fir.global @_QMm1ECb
// CHECK: fir.global @_QMm2ECc
// CHECK: gpu.module @cuda_device_mod
// CHECK-DAG: fir.global @_QMm2ECc
// CHECK-DAG: fir.global @_QMm1ECb
