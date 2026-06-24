// RUN: %clang_cc1 -no-enable-noundef-analysis %s -cl-std=CL2.0 -triple amdgcn -fcommon -O0 -emit-llvm -o -                                         | FileCheck %s
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -cl-std=CL2.0 -triple amdgcn -fcommon -O0 -emit-llvm -o - -fexperimental-new-constant-interpreter | FileCheck %s

// CHECK: @fold_int_local ={{.*}} addrspace(1) global i32 13, align 4
int fold_int_local = (int)(local void*)(generic char*)(global int*)0 + 14;

// CHECK: @fold_int ={{.*}} addrspace(1) global i32 13, align 4
int fold_int = (int)(private void*)(generic char*)(global int*)0 + 14;

// CHECK: @test_static_var_private.sp4 = internal addrspace(1) global ptr addrspace(5) null, align 4
// CHECK: @test_static_var_private.sp5 = internal addrspace(1) global ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)), align 4

void test_static_var_private(void) {
  static private char *sp4 = (private char*)((void)0, 0);
  const int x = 0;
  static private char *sp5 = (private char*)x;
}
