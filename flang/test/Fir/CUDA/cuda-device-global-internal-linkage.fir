// RUN: fir-opt --cuf-device-global %s | FileCheck %s

// Verify that a registered CUDA global with internal host linkage is cloned
// into the gpu.module without `internal`, so runtime symbol lookup can resolve
// the device symbol.
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module} {
  fir.global internal @_QMmtestsEpriv(dense<[1, 2, 3]> : tensor<3xi32>) {data_attr = #cuf.cuda<device>} : !fir.array<3xi32>

  gpu.module @cuda_device_mod {
  }
}

// Host-side linkage is preserved.
// CHECK: fir.global internal @_QMmtestsEpriv(dense<[1, 2, 3]> : tensor<3xi32>) {data_attr = #cuf.cuda<device>} : !fir.array<3xi32>

// GPU clone must not keep `internal`.
// CHECK-LABEL: gpu.module @cuda_device_mod
// CHECK: fir.global @_QMmtestsEpriv(dense<[1, 2, 3]> : tensor<3xi32>) {data_attr = #cuf.cuda<device>} : !fir.array<3xi32>
// CHECK-NOT: fir.global internal @_QMmtestsEpriv
