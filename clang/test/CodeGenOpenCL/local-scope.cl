// RUN: %clang_cc1 %s -triple spir64 -disable-llvm-passes -emit-llvm -o - | FileCheck %s

#pragma OPENCL EXTENSION __cl_clang_function_scope_local_variables : enable

void func(local int*);

void bar() {
  // CHECK: @bar.i = internal addrspace(3) global i32 undef, align 4
  local int i;
  func(&i);
}

__kernel void foo(void) {
  // CHECK: @foo.i = internal addrspace(3) global i32 undef, align 4
  {
    local int i;
    func(&i);
  }
}
