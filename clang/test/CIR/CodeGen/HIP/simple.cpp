#include "../Inputs/cuda.h"

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s


// This should emit as a normal C++ function.
__host__ void host_fn(int *a, int *b, int *c) {}

// CIR: cir.func @_Z7host_fnPiS_S_

// This shouldn't emit.
__device__ void device_fn(int* a, double b, float c) {}

// CHECK-NOT: cir.func @_Z9device_fnPidf
