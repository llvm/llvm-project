// RUN: %clang_cc1 %s -fclangir -triple spirv64-unknown-unknown -emit-cir -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 %s -fclangir -triple spirv64-unknown-unknown -emit-llvm -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM -DHINT_VALUE=undef
// RUN: %clang_cc1 %s -triple spirv64-unknown-unknown -emit-llvm -o %t.ogcg.ll
// RUN: FileCheck %s --input-file=%t.ogcg.ll --check-prefix=LLVM -DHINT_VALUE=poison

typedef unsigned int uint4 __attribute__((ext_vector_type(4)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef float float4 __attribute__((ext_vector_type(4)));

kernel __attribute__((vec_type_hint(int)))
__attribute__((reqd_work_group_size(1, 2, 4))) void kernel1(int a) {}

// CIR-LABEL: cir.func @kernel1
// CIR-SAME: cir.cl.kernel_execution_metadata = #cir.cl.kernel_execution_metadata
// CIR-SAME: reqd_work_group_size = [1, 2, 4]
// CIR-SAME: vec_type_hint = !s32i
// LLVM-LABEL: define spir_kernel void @kernel1
// LLVM-SAME: !vec_type_hint ![[KERNEL1_VEC_TYPE_HINT:[0-9]+]]
// LLVM-SAME: !reqd_work_group_size ![[KERNEL1_REQD_WORK_GROUP_SIZE:[0-9]+]]

kernel __attribute__((vec_type_hint(uint4)))
__attribute__((work_group_size_hint(8, 16, 4294967294U))) void kernel2(int a) {}

// CIR-LABEL: cir.func @kernel2
// CIR-SAME: cir.cl.kernel_execution_metadata = #cir.cl.kernel_execution_metadata
// The unsigned dimension retains its 32-bit representation through LLVM IR.
// CIR-SAME: work_group_size_hint = [8, 16, -2]
// CIR-SAME: vec_type_hint = !cir.vector<4 x !u32i>
// LLVM-LABEL: define spir_kernel void @kernel2
// LLVM-SAME: !vec_type_hint ![[KERNEL2_VEC_TYPE_HINT:[0-9]+]]
// LLVM-SAME: !work_group_size_hint ![[KERNEL2_WORK_GROUP_SIZE_HINT:[0-9]+]]

kernel __attribute__((vec_type_hint(unsigned int))) void unsigned_scalar() {}

// CIR-LABEL: cir.func @unsigned_scalar
// CIR-SAME: vec_type_hint = !u32i
// LLVM-LABEL: define spir_kernel void @unsigned_scalar
// LLVM-SAME: !vec_type_hint ![[UNSIGNED_SCALAR:[0-9]+]]

kernel __attribute__((vec_type_hint(int4))) void signed_vector() {}

// CIR-LABEL: cir.func @signed_vector
// CIR-SAME: vec_type_hint = !cir.vector<4 x !s32i>
// LLVM-LABEL: define spir_kernel void @signed_vector
// LLVM-SAME: !vec_type_hint ![[SIGNED_VECTOR:[0-9]+]]

kernel __attribute__((vec_type_hint(float4))) void float_vector() {}

// CIR-LABEL: cir.func @float_vector
// CIR-SAME: vec_type_hint = !cir.vector<4 x !cir.float>
// LLVM-LABEL: define spir_kernel void @float_vector
// LLVM-SAME: !vec_type_hint ![[FLOAT_VECTOR:[0-9]+]]

void helper(void) {}

// CIR-LABEL: cir.func @helper
// CIR-NOT: cir.cl.kernel_execution_metadata
// LLVM-LABEL: define{{.*}} void @helper
// LLVM-NOT: !vec_type_hint
// LLVM-NOT: !reqd_work_group_size
// LLVM-NOT: !work_group_size_hint

// LLVM-DAG: ![[KERNEL1_VEC_TYPE_HINT]] = !{i32 [[HINT_VALUE]], i32 1}
// LLVM-DAG: ![[KERNEL1_REQD_WORK_GROUP_SIZE]] = !{i32 1, i32 2, i32 4}
// LLVM-DAG: ![[KERNEL2_VEC_TYPE_HINT]] = !{<4 x i32> [[HINT_VALUE]], i32 0}
// LLVM-DAG: ![[KERNEL2_WORK_GROUP_SIZE_HINT]] = !{i32 8, i32 16, i32 -2}
// LLVM-DAG: ![[UNSIGNED_SCALAR]] = !{i32 [[HINT_VALUE]], i32 0}
// LLVM-DAG: ![[SIGNED_VECTOR]] = !{<4 x i32> [[HINT_VALUE]], i32 1}
// LLVM-DAG: ![[FLOAT_VECTOR]] = !{<4 x float> [[HINT_VALUE]], i32 0}
