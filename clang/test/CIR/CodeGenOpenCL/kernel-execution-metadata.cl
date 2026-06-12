// RUN: %clang_cc1 %s -fclangir -triple spirv64-unknown-unknown -emit-cir -o - | FileCheck %s --check-prefix=CIR

typedef unsigned int uint4 __attribute__((ext_vector_type(4)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef float float4 __attribute__((ext_vector_type(4)));

kernel __attribute__((vec_type_hint(int)))
__attribute__((reqd_work_group_size(1, 2, 4))) void kernel1(int a) {}

// CIR-LABEL: cir.func @kernel1
// CIR-SAME: cir.cl.kernel_execution_metadata = #cir.cl.kernel_execution_metadata
// CIR-SAME: reqd_work_group_size = [1, 2, 4]
// CIR-SAME: vec_type_hint = !s32i

kernel __attribute__((vec_type_hint(uint4)))
__attribute__((work_group_size_hint(8, 16, 4294967294U))) void kernel2(int a) {}

// CIR-LABEL: cir.func @kernel2
// CIR-SAME: cir.cl.kernel_execution_metadata = #cir.cl.kernel_execution_metadata
// The unsigned dimension retains its 32-bit representation in CIR.
// CIR-SAME: work_group_size_hint = [8, 16, -2]
// CIR-SAME: vec_type_hint = !cir.vector<4 x !u32i>

kernel __attribute__((vec_type_hint(unsigned int))) void unsigned_scalar() {}

// CIR-LABEL: cir.func @unsigned_scalar
// CIR-SAME: vec_type_hint = !u32i

kernel __attribute__((vec_type_hint(int4))) void signed_vector() {}

// CIR-LABEL: cir.func @signed_vector
// CIR-SAME: vec_type_hint = !cir.vector<4 x !s32i>

kernel __attribute__((vec_type_hint(float4))) void float_vector() {}

// CIR-LABEL: cir.func @float_vector
// CIR-SAME: vec_type_hint = !cir.vector<4 x !cir.float>

void helper(void) {}

// CIR-LABEL: cir.func @helper
// CIR-NOT: cir.cl.kernel_execution_metadata
