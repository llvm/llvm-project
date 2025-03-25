// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1250 -fcuda-is-device -emit-llvm -x hip -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -aux-triple amdgcn-amd-amdhsa -emit-llvm -x hip -o - %s | FileCheck --check-prefix=HOST %s

#include "Inputs/cuda.h"

const int constint = 4;

// HOST-NOT: "amdgpu-cluster-dims"

// CHECK: "amdgpu-cluster-dims"="2,2,2"
__global__ void __cluster_dims__(2, 2, 2) test_literal_3d() {}

// CHECK: "amdgpu-cluster-dims"="2,2,1"
__global__ void __cluster_dims__(2, 2) test_literal_2d() {}

// CHECK: "amdgpu-cluster-dims"="4,1,1"
__global__ void __cluster_dims__(4) test_literal_1d() {}

// CHECK: "amdgpu-cluster-dims"="4,2,1"
__global__ void __cluster_dims__(constint, constint / 2, 1) test_constant() {}

// CHECK: "amdgpu-cluster-dims"="0,0,0"
__global__ void __no_cluster__ test_no_cluster() {}
