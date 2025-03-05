#include "../Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx -fclangir \
// RUN:            -fcuda-is-device -emit-cir -target-sdk-version=12.3 \
// RUN:            %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

__device__ void device_fn(int* a, double b, float c) {}
// CHECK: cir.func @_Z9device_fnPidf
