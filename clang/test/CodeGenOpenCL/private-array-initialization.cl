// RUN: %clang_cc1 %s -triple spir-unknown-unknown -O0 -emit-llvm -o - | FileCheck %s

// CHECK: @test.arr = private unnamed_addr addrspace(2) constant [3 x i32] [i32 1, i32 2, i32 3], align 4

void test() {
  __private int arr[] = {1, 2, 3};
// CHECK:  %[[arr_i8_ptr:[0-9]+]] = bitcast [3 x i32]* %arr to i8*
// CHECK:  call void @llvm.memcpy.p0i8.p2i8.i32(i8* %[[arr_i8_ptr]], i8 addrspace(2)* bitcast ([3 x i32] addrspace(2)* @test.arr to i8 addrspace(2)*), i32 12, i32 4, i1 false)
}
