// RUN: %clang_cc1 -fcuda-is-device -verify=nordc %s
// RUN: %clang_cc1 -fcuda-is-device -fgpu-rdc -verify=rdc %s
// RUN: %clang_cc1 -x hip -fcuda-is-device -verify=hip %s

// rdc-no-diagnostics

#include "Inputs/cuda.h"

__global__ void g2(int x) {}

// CHECK-LABEL: define{{.*}}g1
__global__ void g1(void) {
  // CHECK: [[CONFIG:%.*]] = call{{.*}}_Z22cudaGetParameterBuffermm(i64{{.*}}64, i64{{.*}}4)
  // CHECK-NEXT: [[FLAG:%.*]] = icmp ne ptr [[CONFIG]], null
  // CHECK-NEXT: br i1 [[FLAG]], label %[[THEN:.*]], label %[[ENDIF:.*]]
  // CHECK: [[THEN]]:
  // CHECK-NEXT: [[PPTR:%.*]] = getelementptr{{.*}}i8, ptr [[CONFIG]], i64 0
  // CHECK-NEXT: store i32 42, ptr [[PPTR]]
  // CHECK: = call{{.*}} i32 @_Z16cudaLaunchDevicePvS_4dim3S0_jP10cudaStream(ptr{{.*}} @_Z2g2i, ptr{{.*}} [[CONFIG]],
  g2<<<1, 1>>>(42);
  // nordc-error@-1 {{kernel launch from __device__ or __global__ function requires relocatable device code (i.e. requires -fgpu-rdc)}}
  // hip-error@-2 {{device-side kernel call/launch is not supported}}
}
