// Based on clang/test/CodeGenCUDA/filter-decl.cu tailored for CIR current capabilities.
// Tests that host/device functions are emitted only on the appropriate side.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-sdk-version=9.2 \
// RUN:   -x cuda -emit-cir %s -o %t.host.cir
// RUN: FileCheck --input-file=%t.host.cir %s --check-prefix=CIR-HOST

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda \
// RUN:  -fcuda-is-device -emit-cir %s -o %t.device.cir
// RUN: FileCheck --input-file=%t.device.cir %s --check-prefix=CIR-DEVICE

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-sdk-version=9.2 \
// RUN:   -x cuda -emit-llvm %s -o %t.host.ll
// RUN: FileCheck --input-file=%t.host.ll %s --check-prefix=OGCG-HOST

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda \
// RUN:   -fcuda-is-device -emit-llvm %s -o %t.device.ll
// RUN: FileCheck --input-file=%t.device.ll %s --check-prefix=OGCG-DEVICE

#include "Inputs/cuda.h"

// Implicit host function (no attribute) — host only
// CIR-HOST: cir.func {{.*}} @_Z20implicithostonlyfuncv()
// CIR-DEVICE-NOT: @_Z20implicithostonlyfuncv
// OGCG-HOST: define{{.*}} void @_Z20implicithostonlyfuncv()
// OGCG-DEVICE-NOT: @_Z20implicithostonlyfuncv
void implicithostonlyfunc(void) {}

// Explicit __host__ function — host only
// CIR-HOST: cir.func {{.*}} @_Z20explicithostonlyfuncv()
// CIR-DEVICE-NOT: @_Z20explicithostonlyfuncv
// OGCG-HOST: define{{.*}} void @_Z20explicithostonlyfuncv()
// OGCG-DEVICE-NOT: @_Z20explicithostonlyfuncv
__host__ void explicithostonlyfunc(void) {}

// __device__ function — device only
// CIR-HOST-NOT: @_Z14deviceonlyfuncv
// CIR-DEVICE: cir.func {{.*}} @_Z14deviceonlyfuncv()
// OGCG-HOST-NOT: @_Z14deviceonlyfuncv
// OGCG-DEVICE: define{{.*}} void @_Z14deviceonlyfuncv()
__device__ void deviceonlyfunc(void) {}

// __host__ __device__ function — both sides
// CIR-HOST: cir.func {{.*}} @_Z14hostdevicefuncv()
// CIR-DEVICE: cir.func {{.*}} @_Z14hostdevicefuncv()
// OGCG-HOST: define{{.*}} void @_Z14hostdevicefuncv()
// OGCG-DEVICE: define{{.*}} void @_Z14hostdevicefuncv()
__host__ __device__ void hostdevicefunc(void) {}

// __global__ kernel — both sides (stub on host, kernel on device)
// CIR-HOST: cir.func {{.*}} @_Z25__device_stub__globalfuncv()
// CIR-DEVICE: cir.func {{.*}} @_Z10globalfuncv()
// OGCG-HOST: define{{.*}} void @_Z25__device_stub__globalfuncv()
// OGCG-DEVICE: define{{.*}} void @_Z10globalfuncv()
__global__ void globalfunc(void) {}
