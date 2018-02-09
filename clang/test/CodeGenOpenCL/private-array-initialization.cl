// RUN: %clang_cc1 %s -triple spir-unknown-unknown -O0 -emit-llvm -o - | FileCheck -check-prefix=PRIVATE0 %s
// RUN: %clang_cc1 %s -triple amdgcn-amd-amdhsa-unknown -O0 -emit-llvm -o - | FileCheck -check-prefix=PRIVATE5 %s

// CHECK: @test.arr = private unnamed_addr addrspace(2) constant [3 x i32] [i32 1, i32 2, i32 3], align 4

void test() {
  __private int arr[] = {1, 2, 3};
// PRIVATE0:  %[[arr_i8_ptr:[0-9]+]] = bitcast [3 x i32]* %arr to i8*
// PRIVATE0:  call void @llvm.memcpy.p0i8.p2i8.i32(i8* align 4 %[[arr_i8_ptr]], i8 addrspace(2)* align 4 bitcast ([3 x i32] addrspace(2)* @test.arr to i8 addrspace(2)*), i32 12, i1 false)

// PRIVATE5: %arr = alloca [3 x i32], align 4, addrspace(5)
// PRIVATE5: %0 = bitcast [3 x i32] addrspace(5)* %arr to i8 addrspace(5)*
// PRIVATE5: call void @llvm.memcpy.p5i8.p2i8.i64(i8 addrspace(5)* align 4 %0, i8 addrspace(2)* align 4 bitcast ([3 x i32] addrspace(2)* @test.arr to i8 addrspace(2)*), i64 12, i1 false)
}

__kernel void initializer_cast_is_valid_crash() {
// PRIVATE0: %v512 = alloca [64 x i8], align 1
// PRIVATE0: %0 = bitcast [64 x i8]* %v512 to i8*
// PRIVATE0: call void @llvm.memset.p0i8.i32(i8* align 1 %0, i8 0, i32 64, i1 false)
// PRIVATE0: %1 = bitcast i8* %0 to [64 x i8]*


// PRIVATE5: %v512 = alloca [64 x i8], align 1, addrspace(5)
// PRIVATE5: %0 = bitcast [64 x i8] addrspace(5)* %v512 to i8 addrspace(5)*
// PRIVATE5: call void @llvm.memset.p5i8.i64(i8 addrspace(5)* align 1 %0, i8 0, i64 64, i1 false)
// PRIVATE5: %1 = bitcast i8 addrspace(5)* %0 to [64 x i8] addrspace(5)*
  unsigned char v512[64] = {
      0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
      0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
      0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
      0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x02,0x00
  };
}
