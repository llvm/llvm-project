// RUN: %clang_cc1 %s -triple spir-unknown-unknown -O0 -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-LLVM
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -O0 -emit-llvm-bc -o %t.bc
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

// CHECK-LLVM: @__const.test.arr = private unnamed_addr addrspace(2) constant [3 x i32] [i32 1, i32 2, i32 3], align 4

// CHECK-SPIRV-DAG: TypeInt [[i32:[0-9]+]] 32 0
// CHECK-SPIRV-DAG: TypeInt [[i8:[0-9]+]] 8 0
// CHECK-SPIRV-DAG: Constant [[i32]] [[one:[0-9]+]] 1
// CHECK-SPIRV-DAG: Constant [[i32]] [[two:[0-9]+]] 2
// CHECK-SPIRV-DAG: Constant [[i32]] [[three:[0-9]+]] 3
// CHECK-SPIRV-DAG: Constant [[i32]] [[twelve:[0-9]+]] 12
// CHECK-SPIRV-DAG: TypeArray [[i32x3:[0-9]+]] [[i32]] [[three]]
// CHECK-SPIRV-DAG: TypePointer [[i32x3_ptr:[0-9]+]] 7 [[i32x3]]
// CHECK-SPIRV-DAG: TypePointer [[const_i32x3_ptr:[0-9]+]] 0 [[i32x3]]
// CHECK-SPIRV-DAG: TypePointer [[i8_ptr:[0-9]+]] 7 [[i8]]
// CHECK-SPIRV-DAG: TypePointer [[const_i8_ptr:[0-9]+]] 0 [[i8]]
// CHECK-SPIRV: ConstantComposite [[i32x3]] [[test_arr_init:[0-9]+]] [[one]] [[two]] [[three]]
// CHECK-SPIRV: Variable [[const_i32x3_ptr]] [[test_arr:[0-9]+]] 0 [[test_arr_init]]
// CHECK-SPIRV: Variable [[const_i32x3_ptr]] [[test_arr2:[0-9]+]] 0 [[test_arr_init]]

void test() {
  __private int arr[] = {1,2,3};
  __private const int arr2[] = {1,2,3};
// CHECK-LLVM:  %arr = alloca [3 x i32], align 4
// CHECK-LLVM:  %[[arr_i8_ptr:[0-9]+]] = bitcast [3 x i32]* %arr to i8*
// CHECK-LLVM:  call void @llvm.memcpy.p0i8.p2i8.i32(i8* align 4 %0, i8 addrspace(2)* align 4 bitcast ([3 x i32] addrspace(2)* @__const.test.arr to i8 addrspace(2)*), i32 12, i1 false)

// CHECK-SPIRV: Variable [[i32x3_ptr]] [[arr:[0-9]+]] 7
// CHECK-SPIRV: Variable [[i32x3_ptr]] [[arr2:[0-9]+]] 7

// CHECK-SPIRV: Bitcast [[i8_ptr]] [[arr_i8_ptr:[0-9]+]] [[arr]]
// CHECK-SPIRV: Bitcast [[const_i8_ptr]] [[test_arr_const_i8_ptr:[0-9]+]] [[test_arr]]
// CHECK-SPIRV: CopyMemorySized [[arr_i8_ptr]] [[test_arr_const_i8_ptr]] [[twelve]] 2 4

// CHECK-SPIRV: Bitcast [[i8_ptr]] [[arr2_i8_ptr:[0-9]+]] [[arr2]]
// CHECK-SPIRV: Bitcast [[const_i8_ptr]] [[test_arr2_const_i8_ptr:[0-9]+]] [[test_arr2]]
// CHECK-SPIRV: CopyMemorySized [[arr2_i8_ptr]] [[test_arr2_const_i8_ptr]] [[twelve]] 2 4
}
