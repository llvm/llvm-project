// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple spir64-unkown-unkown -emit-llvm %s -o -| FileCheck %s
// expected-no-diagnostics

typedef int (^block_t)();

int block_typedef_kernel(global int* res) {
  // CHECK: %{{.*}} = alloca <{ i32, i32, i8 addrspace(4)*, [3 x i32] }>
  int a[3] = {1, 2, 3};
  // CHECK: call void @llvm.memcpy{{.*}}
  block_t b = ^() { return a[0]; };
  return b();
}

// CHECK: define {{.*}} @__block_typedef_kernel_block_invoke
// CHECK: %{{.*}} = getelementptr inbounds [3 x i32], [3 x i32] addrspace(4)* %{{.*}}, i64 0, i64 0
// CHECK-NOT: call void @llvm.memcpy{{.*}}
