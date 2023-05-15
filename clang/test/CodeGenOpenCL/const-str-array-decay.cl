// RUN: %clang_cc1 %s -emit-llvm -o - -ffake-address-space-map | FileCheck %s

int test_func(constant char* foo);

kernel void str_array_decy() {
  test_func("Test string literal");
}

// CHECK: call i32 @test_func(ptr addrspace(2) noundef @{{.*}})
