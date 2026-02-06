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

// CHECK: "amdgpu-cluster-dims"="7,1,1"
template<unsigned a>
__global__ void __cluster_dims__(a) test_template_1d() {}
template __global__ void test_template_1d<7>();

// CHECK: "amdgpu-cluster-dims"="2,6,1"
template<unsigned a, unsigned b>
__global__ void __cluster_dims__(a, b) test_template_2d() {}
template __global__ void test_template_2d<2, 6>();

// CHECK: "amdgpu-cluster-dims"="1,2,3"
template<unsigned a, unsigned b, unsigned c>
__global__ void __cluster_dims__(a, b, c) test_template_3d() {}
template __global__ void test_template_3d<1, 2, 3>();
