
// RUN: fir-opt --fir-to-llvm-ir %s | FileCheck %s

gpu.module @cuda_device_mod {
// CHECK-LABEL: llvm.func @allocmem_array_aligned_device
// CHECK: llvm.call @malloc(%{{.*}}) {{.*}} : (i64) -> !llvm.ptr
// CHECK-NOT: @aligned_alloc
// CHECK-NOT: @posix_memalign
  func.func @allocmem_array_aligned_device() -> !fir.heap<!fir.array<1000xf32>> {
    %1 = fir.allocmem !fir.array<1000xf32> {alignment = 64 : i64}
    return %1 : !fir.heap<!fir.array<1000xf32>>
  }
}
