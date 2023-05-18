// RUN: %clang_cc1 %s -triple spir-unknown-unknown -O0 -emit-llvm -o - | FileCheck -check-prefix=PRIVATE0 %s
// RUN: %clang_cc1 %s -triple amdgcn-amd-amdhsa-unknown -O0 -emit-llvm -o - | FileCheck -check-prefix=PRIVATE5 %s

// CHECK: @test.arr = private unnamed_addr addrspace(2) constant [3 x i32] [i32 1, i32 2, i32 3], align 4

void test() {
  __private int arr[] = {1, 2, 3};
// PRIVATE0:  call void @llvm.memcpy.p0.p2.i32(ptr align 4 %arr, ptr addrspace(2) align 4 @__const.test.arr, i32 12, i1 false)

// PRIVATE5: %arr = alloca [3 x i32], align 4, addrspace(5)
// PRIVATE5: call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) align 4 %arr, ptr addrspace(4) align 4 @__const.test.arr, i64 12, i1 false)
}

__kernel void initializer_cast_is_valid_crash() {
// PRIVATE0: %v512 = alloca [64 x i8], align 1
// PRIVATE0: call void @llvm.memset.p0.i32(ptr align 1 %v512, i8 0, i32 64, i1 false)


// PRIVATE5: %v512 = alloca [64 x i8], align 1, addrspace(5)
// PRIVATE5: call void @llvm.memset.p5.i64(ptr addrspace(5) align 1 %v512, i8 0, i64 64, i1 false)
  unsigned char v512[64] = {
      0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
      0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
      0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
      0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x02,0x00
  };
}
