// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -ast-print -x hip -verify=NS,all %s
// RUN: %clang_cc1 -triple nvptx-nvidia-cuda -fcuda-is-device -ast-print -x hip -verify=NS,all %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1250 -fcuda-is-device -ast-print -x hip -verify=amd,common,all %s | FileCheck -check-prefixes=CHECK %s
// RUN: %clang_cc1 -triple nvptx-nvidia-cuda -target-cpu sm_90 -fcuda-is-device -ast-print -x hip -verify=cuda,common,all %s | FileCheck -check-prefixes=CHECK %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -aux-triple amdgcn-amd-amdhsa -ast-print -x hip -verify=amd,common,all %s | FileCheck -check-prefixes=CHECK %s

#include "Inputs/cuda.h"

const int constint = 4;

// CHECK: __attribute__((global)) __attribute__((cluster_dims(2, 2, 2))) void test_literal_3d()
__global__ void __cluster_dims__(2, 2, 2) test_literal_3d() {} //NS-error {{'cluster_dims' is not supported for this GPU architecture}}

// CHECK: __attribute__((global)) __attribute__((cluster_dims(2, 2))) void test_literal_2d()
__global__ void __cluster_dims__(2, 2) test_literal_2d() {} //NS-error {{'cluster_dims' is not supported for this GPU architecture}}

// CHECK: __attribute__((global)) __attribute__((cluster_dims(4))) void test_literal_1d()
__global__ void __cluster_dims__(4) test_literal_1d() {} //NS-error {{'cluster_dims' is not supported for this GPU architecture}}

// CHECK: __attribute__((global)) __attribute__((cluster_dims(constint, constint / 4, 1))) void test_constant()
__global__ void __cluster_dims__(constint, constint / 4, 1) test_constant() {} //NS-error {{'cluster_dims' is not supported for this GPU architecture}}

// CHECK: template <int x, int y, int z> void test_template() __attribute__((cluster_dims(x, y, z)))
template <int x, int y, int z>  void test_template(void) __cluster_dims__(x, y, z){} //NS-error {{'cluster_dims' is not supported for this GPU architecture}}

// CHECK: template <int x, int y, int z> void test_template_expr() __attribute__((cluster_dims(x + constint, y, z)))
template <int x, int y, int z> void test_template_expr(void) __cluster_dims__(x + constint, y, z) {} //NS-error {{'cluster_dims' is not supported for this GPU architecture}}

//NS-error@+1 {{'cluster_dims' is not supported for this GPU architecture}}
__global__ void __cluster_dims__(32, 2, 4) test_too_large_dim_0() {} // common-error {{integer constant expression evaluates to value 32 that cannot be represented in a 4-bit unsigned integer type}}

// cuda-error@+2 {{cluster does not support more than 8 thread blocks; 64 provided}}
// amd-error@+1 {{cluster does not support more than 16 thread blocks; 64 provided}}
__global__ void __cluster_dims__(4, 4, 4) test_too_large_dim_1() {} // NS-error {{'cluster_dims' is not supported for this GPU architecture}}

// cuda-error@+3 {{cluster does not support more than 8 thread blocks; 64 provided}}
// amd-error@+2 {{cluster does not support more than 16 thread blocks; 64 provided}}
template<unsigned a, unsigned b, unsigned c>
__global__ void __cluster_dims__(a, b, c) test_too_large_dim_template() {} // NS-error {{'cluster_dims' is not supported for this GPU architecture}}
template __global__ void test_too_large_dim_template<4, 4, 4>(); // common-note {{in instantiation of function template specialization 'test_too_large_dim_template<4U, 4U, 4U>' requested here}}

int none_const_int = 4;

//NS-error@+1 {{'cluster_dims' is not supported for this GPU architecture}}
__global__ void __cluster_dims__(none_const_int, 2, 4) test_non_constant_0() {} // common-error {{'cluster_dims' attribute requires parameter 0 to be an integer constant}}

//NS-error@+1 {{'cluster_dims' is not supported for this GPU architecture}}
__global__ void __cluster_dims__(8, none_const_int / 2, 4) test_non_constant_1() {} // common-error {{'cluster_dims' attribute requires parameter 1 to be an integer constant}}

//NS-error@+1 {{'cluster_dims' is not supported for this GPU architecture}}
__global__ void __cluster_dims__(8, 2, none_const_int / 4) test_non_constant_2() {} // common-error {{'cluster_dims' attribute requires parameter 2 to be an integer constant}}

//NS-error@+1 {{'no_cluster' is not supported for this GPU architecture}}
__global__ void __no_cluster__ test_no_cluster() {}

//NS-error@+2 {{'no_cluster' is not supported for this GPU architecture}}
//NS-error@+1 {{'cluster_dims' is not supported for this GPU architecture}}
__global__ void __no_cluster__ __cluster_dims__(2,2,2) test_have_both() {} // common-error {{'cluster_dims' and 'no_cluster' attributes are not compatible}} common-note {{conflicting attribute is here}}

template <int... args>
__cluster_dims__(args) void test_template_variadic_args(void) {} // all-error {{expression contains unexpanded parameter pack 'args'}}

template <int... args>
__cluster_dims__(1, args) void test_template_variadic_args_2(void) {} // all-error {{expression contains unexpanded parameter pack 'args'}}
