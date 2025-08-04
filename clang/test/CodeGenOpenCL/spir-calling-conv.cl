// RUN: %clang_cc1 %s -triple "spir-unknown-unknown" -emit-llvm -o - | FileCheck %s

int get_dummy_id(int D);

kernel void bar(global int *A);

//CHECK: define{{.*}} spir_kernel void @foo(ptr addrspace(1) noundef align 4 %A)
//CHECK: tail call spir_func void @__clang_ocl_kern_imp_bar(ptr addrspace(1) noundef align 4 %A)

kernel void foo(global int *A)
// CHECK: define{{.*}} spir_func void @__clang_ocl_kern_imp_foo(ptr addrspace(1) noundef align 4 %A)
{
  int id = get_dummy_id(0);
  // CHECK: %{{[a-z0-9_]+}} = tail call spir_func i32 @get_dummy_id(i32 noundef 0)
  A[id] = id;
  bar(A);
  // CHECK: tail call spir_func void @__clang_ocl_kern_imp_bar(ptr addrspace(1) noundef align 4 %A)
}

// CHECK: declare spir_func i32 @get_dummy_id(i32 noundef)
// CHECK: declare spir_func void @__clang_ocl_kern_imp_bar(ptr addrspace(1) noundef align 4)
