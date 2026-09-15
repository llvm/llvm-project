// RUN: tco --target=aarch64-apple-darwin %s | FileCheck %s

// CHECK-LABEL: define internal void @omp_region_aligned..omp_par(
// CHECK: %[[MEMPTR:.*]] = alloca ptr, i64 1
// CHECK: store ptr null, ptr %[[MEMPTR]]
// CHECK: call i32 @posix_memalign(ptr %[[MEMPTR]], i64 64, i64 4000)
func.func @omp_region_aligned() {
  omp.parallel {
    %1 = fir.allocmem !fir.array<1000xf32> {alignment = 64 : i64}
    fir.freemem %1 : !fir.heap<!fir.array<1000xf32>>
    omp.terminator
  }
  return
}
