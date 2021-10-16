// RUN: %clang_cc1 %s -triple "spir-unknown-unknown" -emit-llvm -o - | FileCheck %s

int get_dummy_id(int D);

kernel void bar(global int *A);

kernel void foo(global int *A)
// CHECK: define{{.*}} spir_kernel void @foo(i32 addrspace(1)* noundef %A)
{
  int id = get_dummy_id(0);
  // CHECK: %{{[a-z0-9_]+}} = tail call spir_func i32 @get_dummy_id(i32 noundef 0)
  A[id] = id;
  bar(A);
  // CHECK: tail call spir_kernel void @bar(i32 addrspace(1)* noundef %A)
}

// CHECK: declare spir_func i32 @get_dummy_id(i32 noundef)
// CHECK: declare spir_kernel void @bar(i32 addrspace(1)* noundef)
