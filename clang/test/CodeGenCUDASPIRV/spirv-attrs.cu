// RUN: %clang_cc1 -fcuda-is-device -triple spirv64 -o - -emit-llvm -x cuda %s  | FileCheck %s
// RUN: %clang_cc1 -fcuda-is-device -triple spirv32 -o - -emit-llvm -x cuda %s  | FileCheck %s

#define __global__ __attribute__((global))

__attribute__((reqd_work_group_size(0, 0, 0)))
__global__ void reqd_work_group_size_0_0_0() {}

__attribute__((reqd_work_group_size(128, 1, 1)))
__global__ void reqd_work_group_size_128_1_1() {}

__attribute__((work_group_size_hint(2, 2, 2)))
__global__ void work_group_size_hint_2_2_2() {}

__attribute__((vec_type_hint(int)))
__global__ void vec_type_hint_int() {}

__attribute__((intel_reqd_sub_group_size(64)))
__global__ void intel_reqd_sub_group_size_64() {}

template <unsigned a, unsigned b, unsigned c>
__attribute__((reqd_work_group_size(a, b, c)))
__global__ void reqd_work_group_size_a_b_c() {}

template __global__ void reqd_work_group_size_a_b_c<256,1,1>(void);

// CHECK: define spir_kernel void @_Z26reqd_work_group_size_0_0_0v() #[[ATTR:[0-9]+]] !reqd_work_group_size ![[WG_SIZE_ZEROS:[0-9]+]]
// CHECK: define spir_kernel void @_Z28reqd_work_group_size_128_1_1v() #[[ATTR:[0-9]+]] !reqd_work_group_size ![[WG_SIZE:[0-9]+]]
// CHECK: define spir_kernel void @_Z26work_group_size_hint_2_2_2v() #[[ATTR]] !work_group_size_hint ![[WG_HINT:[0-9]+]]
// CHECK: define spir_kernel void @_Z17vec_type_hint_intv() #[[ATTR]] !vec_type_hint ![[VEC_HINT:[0-9]+]]
// CHECK: define spir_kernel void @_Z28intel_reqd_sub_group_size_64v() #[[ATTR]] !intel_reqd_sub_group_size ![[SUB_GRP:[0-9]+]]
// CHECK: define spir_kernel void @_Z26reqd_work_group_size_a_b_cILj256ELj1ELj1EEvv() #[[ATTR]] comdat !reqd_work_group_size ![[WG_SIZE_TMPL:[0-9]+]]

// CHECK: attributes #[[ATTR]] = { {{.*}} }

// CHECK: ![[WG_SIZE_ZEROS]] = !{i32 0, i32 0, i32 0}
// CHECK: ![[WG_SIZE]] = !{i32 128, i32 1, i32 1}
// CHECK: ![[WG_HINT]] = !{i32 2, i32 2, i32 2}
// CHECK: ![[VEC_HINT]] = !{i32 poison, i32 1}
// CHECK: ![[SUB_GRP]] = !{i32 64}
// CHECK: ![[WG_SIZE_TMPL]] = !{i32 256, i32 1, i32 1}
