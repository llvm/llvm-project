// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -S -emit-llvm -o - %s | FileCheck %s

// CHECK-NOT: %struct.single_element_struct_arg = type { i32 }
typedef struct single_element_struct_arg
{
    int i;
} single_element_struct_arg_t;

// CHECK: %struct.struct_arg = type { i32, float, i32 }
typedef struct struct_arg
{
    int i1;
    float f;
    int i2;
} struct_arg_t;

// CHECK: %struct.struct_of_arrays_arg = type { [2 x i32], float, [4 x i32], [3 x float], i32 }
typedef struct struct_of_arrays_arg
{
    int i1[2];
    float f1;
    int i2[4];
    float f2[3];
    int i3;
} struct_of_arrays_arg_t;

// CHECK: %struct.struct_of_structs_arg = type { i32, float, %struct.struct_arg, i32 }
typedef struct struct_of_structs_arg
{
    int i1;
    float f1;
    struct_arg_t s1;
    int i2;
} struct_of_structs_arg_t;

// CHECK-LABEL: @test_single_element_struct_arg
// CHECK: i32 %arg1.coerce
__kernel void test_single_element_struct_arg(single_element_struct_arg_t arg1)
{
}

// CHECK-LABEL: @test_struct_arg
// CHECK: %struct.struct_arg %arg1.coerce
__kernel void test_struct_arg(struct_arg_t arg1)
{
}

// CHECK-LABEL: @test_struct_of_arrays_arg
// CHECK: %struct.struct_of_arrays_arg %arg1.coerce
__kernel void test_struct_of_arrays_arg(struct_of_arrays_arg_t arg1)
{
}

// CHECK-LABEL: @test_struct_of_structs_arg
// CHECK: %struct.struct_of_structs_arg %arg1.coerce
__kernel void test_struct_of_structs_arg(struct_of_structs_arg_t arg1)
{
}

// CHECK-LABEL: @test_non_kernel_struct_arg
// CHECK-NOT: %struct.struct_arg %arg1.coerce
// CHECK: %struct.struct_arg* byval
void test_non_kernel_struct_arg(struct_arg_t arg1)
{
}
