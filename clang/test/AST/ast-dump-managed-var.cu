// RUN: %clang_cc1 -ast-dump -x hip %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fcuda-is-device -x hip %s | FileCheck %s

#include "Inputs/cuda.h"

// CHECK-LABEL: VarDecl {{.*}} m1 'int'
// CHECK-NEXT: HIPManagedAttr
// CHECK-NEXT: CUDADeviceAttr {{.*}}Implicit
__managed__ int m1;

// CHECK-LABEL: VarDecl {{.*}} m2 'int'
// CHECK-NEXT: HIPManagedAttr
// CHECK-NEXT: CUDADeviceAttr {{.*}}Implicit
// CHECK-NOT: HIPManagedAttr
// CHECK-NOT: CUDADeviceAttr
__managed__ __managed__ int m2;

// CHECK-LABEL: VarDecl {{.*}} m3 'int'
// CHECK-NEXT: HIPManagedAttr
// CHECK-NEXT: CUDADeviceAttr {{.*}}line
// CHECK-NOT: CUDADeviceAttr {{.*}}Implicit
__managed__ __device__ int m3;

// CHECK-LABEL: VarDecl {{.*}} m3a 'int'
// CHECK-NEXT: CUDADeviceAttr {{.*}}cuda.h
// CHECK-NEXT: HIPManagedAttr
// CHECK-NOT: CUDADeviceAttr {{.*}}Implicit
__device__ __managed__ int m3a;
