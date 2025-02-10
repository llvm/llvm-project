#include "../Inputs/cuda.h"

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:            -fclangir -emit-cir -o - %s | FileCheck %s

// This shouldn't emit.
__host__ void host_fn(int *a, int *b, int *c) {}

// CHECK-NOT: cir.func @_Z7host_fnPiS_S_

// This should emit as a normal C++ function.
__device__ void device_fn(int* a, double b, float c) {}

// CIR: cir.func @_Z9device_fnPidf
