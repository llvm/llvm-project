// RUN: %clang_cc1 %s -triple spir-unknown-unknown -O0 -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-LLVM

// CHECK-LLVM: @__const.test.arr = private unnamed_addr addrspace(2) constant [3 x i32] [i32 1, i32 2, i32 3], align 4

void test() {
  __private int arr[] = {1,2,3};
  __private const int arr2[] = {1,2,3};
// CHECK-LLVM:  %arr = alloca [3 x i32], align 4
// CHECK-LLVM:  %[[arr_i8_ptr:[0-9]+]] = bitcast [3 x i32]* %arr to i8*
// CHECK-LLVM:  call void @llvm.memcpy.p0i8.p2i8.i32(i8* align 4 %[[arr_i8_ptr]], i8 addrspace(2)* align 4 bitcast ([3 x i32] addrspace(2)* @__const.test.arr to i8 addrspace(2)*), i32 12, i1 false)

}
