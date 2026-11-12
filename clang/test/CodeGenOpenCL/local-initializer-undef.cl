// RUN: %clang_cc1 %s -O0 -ffake-address-space-map -emit-llvm -o - | FileCheck %s

typedef struct Foo {
    int x;
    float y;
    float z;
} Foo;

// CHECK-DAG: @test.lds_int = internal addrspace(3) global i32 undef
// CHECK-DAG: @test.lds_int_arr = internal addrspace(3) global [128 x i32] undef
// CHECK-DAG: @test.lds_struct = internal addrspace(3) global %struct.Foo undef
// CHECK-DAG: @test.lds_struct_arr = internal addrspace(3) global [64 x %struct.Foo] undef
__kernel void test()
{
    __local int lds_int;
    __local int lds_int_arr[128];
    __local Foo lds_struct;
    __local Foo lds_struct_arr[64];

    lds_int = 1;
    lds_int_arr[0] = 1;
    lds_struct.x = 1;
    lds_struct_arr[0].x = 1;
}
