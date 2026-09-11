// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux-gnu \
// RUN:   -fopenmp-targets=amdgpu-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c -triple amdgpu-amd-amdhsa \
// RUN:   -fopenmp-targets=amdgpu-amd-amdhsa -emit-llvm %s \
// RUN:   -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - \
// RUN:   | FileCheck %s

// expected-no-diagnostics

// Check that the value returned by the warp shuffle is narrowed back to the
// width of the reduction element before it is stored into the reduction slot.
// The shuffle runtime functions always return a 32- or 64-bit value, so storing
// it unnarrowed writes past the end of a slot for a narrower element such as
// 'half'.

_Float16 half_reduction(_Float16 *a, int n) {
  _Float16 s = 0;
#pragma omp target parallel for map(tofrom : s) reduction(+ : s)
  for (int i = 0; i < n; ++i)
    s += a[i];
  return s;
}

// CHECK-LABEL: define internal void @_omp_reduction_shuffle_and_reduce_func(
// CHECK: %[[ELEM:.+]] = alloca half, align 2
// CHECK: %[[SHUFFLE:.+]] = call i32 @__kmpc_shuffle_int32(
// CHECK-NEXT: %[[NARROWED:.+]] = trunc i32 %[[SHUFFLE]] to i16
// AMDGPU allocas live in addrspace(5) and are accessed through a cast to the
// generic address space, so the store goes through '<name>.ascast' there.
// CHECK-NEXT: store i16 %[[NARROWED]], ptr %[[ELEM]]{{(\.ascast)?}}, align 2
