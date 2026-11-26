// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir
// RUN: FileCheck %s --check-prefix=NEG --input-file=%t.cir
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir
// RUN: FileCheck %s --check-prefix=NEG --input-file=%t.cir

#include "Inputs/cuda.h"

// Device variables ODR-used only by host code still need device definitions.
// CHECK-DAG: cir.global {{.+}} @_ZL10device_var =
static __device__ int device_var;
// CHECK-DAG: cir.global {{.+}} @_ZL12constant_var =
static __constant__ int constant_var;

// NEG-NOT: unused_var
static __device__ int unused_var;

// NEG-NOT: getDeviceSymbol
void getDeviceSymbol(const void *);

// NEG-NOT: host_only
void host_only() {
  getDeviceSymbol(&device_var);
  getDeviceSymbol(&constant_var);
  (void)sizeof(unused_var);
}
