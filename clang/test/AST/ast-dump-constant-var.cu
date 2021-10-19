// RUN: %clang_cc1 -std=c++14 -ast-dump -x hip %s | FileCheck -check-prefixes=CHECK,HOST %s
// RUN: %clang_cc1 -std=c++14 -ast-dump -fcuda-is-device -x hip %s | FileCheck -check-prefixes=CHECK,DEV %s

#include "Inputs/cuda.h"

// CHECK-LABEL: VarDecl {{.*}} m1 'int'
// CHECK-NEXT: CUDAConstantAttr {{.*}}cuda.h
__constant__ int m1;

// CHECK-LABEL: VarDecl {{.*}} m2 'int'
// CHECK-NEXT: CUDAConstantAttr {{.*}}cuda.h
// CHECK-NOT: CUDAConstantAttr
__constant__ __constant__ int m2;

// CHECK-LABEL: VarDecl {{.*}} m3 'const int'
// HOST-NOT: CUDAConstantAttr
// DEV-NOT: CUDAConstantAttr {{.*}}cuda.h
// DEV: CUDAConstantAttr {{.*}}Implicit
// DEV-NOT: CUDAConstantAttr {{.*}}cuda.h
constexpr int m3 = 1;

// CHECK-LABEL: VarDecl {{.*}} m3a 'const int'
// CHECK-NOT: CUDAConstantAttr {{.*}}Implicit
// CHECK: CUDAConstantAttr {{.*}}cuda.h
// CHECK-NOT: CUDAConstantAttr {{.*}}Implicit
constexpr __constant__ int m3a = 2;

// CHECK-LABEL: VarDecl {{.*}} m3b 'const int'
// CHECK-NOT: CUDAConstantAttr {{.*}}Implicit
// CHECK: CUDAConstantAttr {{.*}}cuda.h
// CHECK-NOT: CUDAConstantAttr {{.*}}Implicit
__constant__ constexpr int m3b = 3;
