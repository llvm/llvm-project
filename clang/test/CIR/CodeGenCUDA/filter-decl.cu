// Based on clang/test/CodeGenCUDA/filter-decl.cu tailored for CIR current capabilities.
// Tests that host/device functions are emitted only on the appropriate side.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x cuda \
// RUN:   -I%S/../inputs -emit-cir %s -o %t.host.cir
// RUN: FileCheck --input-file=%t.host.cir %s --check-prefix=CHECK-HOST

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda \
// RUN:   -I%S/../inputs -fcuda-is-device -emit-cir %s -o %t.device.cir
// RUN: FileCheck --input-file=%t.device.cir %s --check-prefix=CHECK-DEVICE

#include "cuda.h"

// Implicit host function (no attribute) — host only
// CHECK-HOST: cir.func {{.*}} @_Z20implicithostonlyfuncv()
// CHECK-DEVICE-NOT: @_Z20implicithostonlyfuncv
void implicithostonlyfunc(void) {}

// Explicit __host__ function — host only
// CHECK-HOST: cir.func {{.*}} @_Z20explicithostonlyfuncv()
// CHECK-DEVICE-NOT: @_Z20explicithostonlyfuncv
__host__ void explicithostonlyfunc(void) {}

// __device__ function — device only
// CHECK-HOST-NOT: @_Z14deviceonlyfuncv
// CHECK-DEVICE: cir.func {{.*}} @_Z14deviceonlyfuncv()
__device__ void deviceonlyfunc(void) {}

// __host__ __device__ function — both sides
// CHECK-HOST: cir.func {{.*}} @_Z14hostdevicefuncv()
// CHECK-DEVICE: cir.func {{.*}} @_Z14hostdevicefuncv()
__host__ __device__ void hostdevicefunc(void) {}

// __global__ kernel — both sides (stub on host, kernel on device)
// CHECK-HOST: cir.func {{.*}} @__device_stub__globalfunc()
// CHECK-DEVICE: cir.func {{.*}} @_Z10globalfuncv()
__global__ void globalfunc(void) {}
