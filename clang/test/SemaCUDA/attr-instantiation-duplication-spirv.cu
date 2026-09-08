// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -fcuda-is-device -x hip -ast-dump -ast-dump-filter test_ %s | FileCheck %s

#include "Inputs/cuda.h"

// CHECK: FunctionDecl {{.*}} test_reqd_work_group_size 'void ()' explicit_instantiation_definition
// CHECK-NEXT: TemplateArgument integral '4'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReqdWorkGroupSizeAttr
// CHECK-NEXT: SubstNonTypeTemplateParmExpr
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}} N
// CHECK-NEXT: IntegerLiteral {{.*}} 4
// CHECK-NEXT: IntegerLiteral {{.*}} 1
// CHECK-NEXT: IntegerLiteral {{.*}} 1
// CHECK-NEXT: CUDAGlobalAttr
// CHECK-EMPTY:
template <int N>
__attribute__((reqd_work_group_size(N, 1, 1)))
__global__ void test_reqd_work_group_size() {}
template __global__ void test_reqd_work_group_size<4>();

template <typename T>
__global__ void test_explicit_specialization() {}

// CHECK: FunctionDecl {{.*}} test_explicit_specialization 'void ()' explicit_specialization
// CHECK-NEXT: TemplateArgument type 'int'
// CHECK-NEXT: BuiltinType {{.*}} 'int'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CUDAGlobalAttr
// CHECK-EMPTY:
template <>
__global__ void test_explicit_specialization<int>() {}
