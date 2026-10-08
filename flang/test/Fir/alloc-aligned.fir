// RUN: tco --target=x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefixes=CHECK,POSIX
// RUN: tco --target=aarch64-apple-darwin %s | FileCheck %s --check-prefixes=CHECK,DARWIN
// RUN: tco --target=nvptx64-nvidia-cuda %s | FileCheck %s --check-prefixes=CHECK,FALLBACK
// RUN: tco --target=amdgcn-amd-amdhsa %s | FileCheck %s --check-prefixes=CHECK,FALLBACK
// RUN: tco --target=x86_64-pc-windows-msvc %s | FileCheck %s --check-prefixes=CHECK,FALLBACK

// CHECK-LABEL: define ptr @allocmem_array_aligned(
// POSIX: call ptr @aligned_alloc(i64 64, i64 4032)
// DARWIN: %[[MEMPTR:.*]] = alloca ptr, i64 1
// DARWIN: store ptr null, ptr %[[MEMPTR]]
// DARWIN: %{{.*}} = call i32 @posix_memalign(ptr %[[MEMPTR]], i64 64, i64 4000)
// DARWIN: %{{.*}} = load ptr, ptr %[[MEMPTR]]
// FALLBACK: call ptr @malloc(i64 4000)
func.func @allocmem_array_aligned() -> !fir.heap<!fir.array<1000xf32>> {
  %1 = fir.allocmem !fir.array<1000xf32> {alignment = 64 : i64}
  return %1 : !fir.heap<!fir.array<1000xf32>>
}

// CHECK-LABEL: define ptr @allocmem_array_dynamic_aligned(
// DARWIN: %[[MEMPTR:.*]] = alloca ptr, i64 1
// CHECK: %[[SIZE:.*]] = mul i64 4, %{{.*}}
// CHECK: %[[NONZERO:.*]] = select i1 %{{.*}}, i64 %[[SIZE]], i64 1
// POSIX: %[[PLUS:.*]] = add i64 %[[NONZERO]], 63
// POSIX: %[[ROUNDED:.*]] = and i64 %[[PLUS]], -64
// POSIX: call ptr @aligned_alloc(i64 64, i64 %[[ROUNDED]])
// DARWIN: store ptr null, ptr %[[MEMPTR]]
// DARWIN: %{{.*}} = call i32 @posix_memalign(ptr %[[MEMPTR]], i64 64, i64 %[[NONZERO]])
// DARWIN: %{{.*}} = load ptr, ptr %[[MEMPTR]]
// FALLBACK: call ptr @malloc(i64 %[[NONZERO]])
func.func @allocmem_array_dynamic_aligned(%n : index) -> !fir.heap<!fir.array<?xf32>> {
  %1 = fir.allocmem !fir.array<?xf32>, %n {alignment = 64 : i64}
  return %1 : !fir.heap<!fir.array<?xf32>>
}
