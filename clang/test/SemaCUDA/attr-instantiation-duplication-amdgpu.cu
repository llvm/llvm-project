// RUN: %clang_cc1 -triple amdgpu12.50-amd-amdhsa -fcuda-is-device -x hip -ast-dump -ast-dump-filter test_ %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -aux-triple amdgpu-amd-amdhsa -x hip -ast-dump -ast-dump-filter test_host_stub %s | FileCheck --check-prefix=HOST %s

#include "Inputs/cuda.h"

// CHECK: FunctionDecl {{.*}} test_flat_work_group_size 'void ()' explicit_instantiation_definition
// CHECK-NEXT: TemplateArgument integral '256'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: AMDGPUFlatWorkGroupSizeAttr
// CHECK-NEXT: IntegerLiteral {{.*}} 1
// CHECK-NEXT: SubstNonTypeTemplateParmExpr
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}} N
// CHECK-NEXT: IntegerLiteral {{.*}} 256
// CHECK-NEXT: CUDAGlobalAttr
// CHECK-EMPTY:
template <int N>
__attribute__((amdgpu_flat_work_group_size(1, N)))
__global__ void test_flat_work_group_size() {}
template __global__ void test_flat_work_group_size<256>();

// CHECK: FunctionDecl {{.*}} test_waves_per_eu 'void ()' explicit_instantiation_definition
// CHECK-NEXT: TemplateArgument integral '2'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: AMDGPUWavesPerEUAttr
// CHECK-NEXT: SubstNonTypeTemplateParmExpr
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}} N
// CHECK-NEXT: IntegerLiteral {{.*}} 2
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: CUDAGlobalAttr
// CHECK-EMPTY:
template <int N>
__attribute__((amdgpu_waves_per_eu(N)))
__global__ void test_waves_per_eu() {}
template __global__ void test_waves_per_eu<2>();

// CHECK: FunctionDecl {{.*}} test_max_num_work_groups 'void ()' explicit_instantiation_definition
// CHECK-NEXT: TemplateArgument integral '8'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: AMDGPUMaxNumWorkGroupsAttr
// CHECK-NEXT: SubstNonTypeTemplateParmExpr
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}} N
// CHECK-NEXT: IntegerLiteral {{.*}} 8
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: CUDAGlobalAttr
// CHECK-EMPTY:
template <int N>
__attribute__((amdgpu_max_num_work_groups(N)))
__global__ void test_max_num_work_groups() {}
template __global__ void test_max_num_work_groups<8>();

// CHECK: FunctionDecl {{.*}} test_cluster_dims 'void ()' explicit_instantiation_definition
// CHECK-NEXT: TemplateArgument integral '4'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CUDAClusterDimsAttr
// CHECK-NEXT: ConstantExpr
// CHECK-NEXT: value: Int 4
// CHECK-NEXT: SubstNonTypeTemplateParmExpr
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}} N
// CHECK-NEXT: IntegerLiteral {{.*}} 4
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: CUDAGlobalAttr
// CHECK-EMPTY:
template <int N>
__attribute__((cluster_dims(N)))
__global__ void test_cluster_dims() {}
template __global__ void test_cluster_dims<4>();

// HOST: FunctionDecl {{.*}} test_host_stub 'void ()' explicit_instantiation_definition
// HOST-NEXT: TemplateArgument integral '4'
// HOST-NEXT: CompoundStmt
// HOST-NEXT: CUDAGlobalAttr
// HOST-NEXT: NoDebugAttr {{.*}} Implicit
// HOST-EMPTY:
template <int N>
__global__ void test_host_stub() {}
template __global__ void test_host_stub<4>();
