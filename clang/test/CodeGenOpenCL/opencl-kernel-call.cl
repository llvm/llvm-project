// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -emit-llvm -o - %s | FileCheck %s

// CHECK: define dso_local amdgpu_kernel void @callee_kern({{.*}})
__attribute__((noinline)) kernel void callee_kern(global int *A){
  *A = 1;
}

__attribute__((noinline)) kernel void ext_callee_kern(global int *A);

// CHECK: define dso_local void @callee_func({{.*}})
__attribute__((noinline)) void callee_func(global int *A){
  *A = 2;
}

// CHECK: define dso_local amdgpu_kernel void @caller_kern({{.*}})
kernel void caller_kern(global int* A){
  callee_kern(A);
  // CHECK: tail call void @__clang_ocl_kern_imp_callee_kern({{.*}})
  ext_callee_kern(A);
  // CHECK: tail call void @__clang_ocl_kern_imp_ext_callee_kern({{.*}})
  callee_func(A);
  // CHECK: tail call void @callee_func({{.*}})

}

// CHECK: define dso_local void @__clang_ocl_kern_imp_callee_kern({{.*}})

// CHECK: declare void @__clang_ocl_kern_imp_ext_callee_kern({{.*}})

// CHECK: define dso_local void @caller_func({{.*}})
void caller_func(global int* A){
  callee_kern(A);
  // CHECK: tail call void @__clang_ocl_kern_imp_callee_kern({{.*}}) #7
  ext_callee_kern(A);
  // CHECK: tail call void @__clang_ocl_kern_imp_ext_callee_kern({{.*}}) #8
  callee_func(A);
  // CHECK: tail call void @callee_func({{.*}})
}

// CHECK: define dso_local void @__clang_ocl_kern_imp_caller_kern({{.*}}) 
// CHECK: tail call void @__clang_ocl_kern_imp_callee_kern({{.*}}) 
// CHECK: tail call void @__clang_ocl_kern_imp_ext_callee_kern({{.*}}) 
// CHECK: tail call void @callee_func({{.*}})
