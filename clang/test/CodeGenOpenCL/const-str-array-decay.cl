// RUN: %clang_cc1 %s -emit-llvm -o - -ffake-address-space-map | FileCheck %s

int test_func(constant char* foo);

kernel void str_array_decy() {
  test_func("Test string literal");
}

// CHECK: i8 addrspace(2)* noundef getelementptr inbounds ([20 x i8], [20 x i8] addrspace(2)*
// CHECK-NOT: addrspacecast

