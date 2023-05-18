// RUN: %clang_cc1 %s -triple "spir64-unknown-unknown" -cl-opt-disable -ffake-address-space-map -emit-llvm -o - | FileCheck %s

// CHECK: @array ={{.*}} addrspace({{[0-9]+}}) constant
__constant float array[2] = {0.0f, 1.0f};

kernel void test(global float *out) {
  *out = array[0];
}

// Test that we don't use directly initializers for const aggregates
// but create a copy in the original address space (unless a variable itself is
// in the constant address space).

void foo(constant int* p, constant const int *p1, const int *p2, const int *p3);
// CHECK: @k.arr1 = internal addrspace(2) constant [3 x i32] [i32 1, i32 2, i32 3]
// CHECK: @__const.k.arr2 = private unnamed_addr addrspace(2) constant [3 x i32] [i32 4, i32 5, i32 6]
// CHECK: @__const.k.arr3 = private unnamed_addr addrspace(2) constant [3 x i32] [i32 7, i32 8, i32 9]
// CHECK: @k.var1 = internal addrspace(2) constant i32 1
kernel void k(void) {
  // CHECK-NOT: %arr1 = alloca [3 x i32]
  constant const int arr1[] = {1, 2, 3};
  // CHECK: %arr2 = alloca [3 x i32]
  const int arr2[] = {4, 5, 6};
  // CHECK: %arr3 = alloca [3 x i32]
  int arr3[] = {7, 8, 9};

  constant int var1 = 1;
  
  // CHECK: call spir_func void @foo(ptr addrspace(2) noundef @k.var1, ptr addrspace(2) noundef @k.arr1
  foo(&var1, arr1, arr2, arr3);
}
