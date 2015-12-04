// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -O0 %s -o - 2>&1 -std=c++11 | FileCheck %s

int gi;

namespace lambdas {
// CHECK-LABEL: define void @_ZN7lambdas7LambdasEPc
void Lambdas(char *ptr) {
  auto L1 = [](void *const p __attribute__((pass_object_size(0)))) {
    return __builtin_object_size(p, 0);
  };

  int i = 0;
  auto L2 = [&i](void *const p __attribute__((pass_object_size(0)))) {
    return __builtin_object_size(p, 0) + i;
  };

  // CHECK: @llvm.objectsize
  gi = L1(ptr);
  // CHECK: @llvm.objectsize
  gi = L2(ptr);
}

// CHECK-DAG: define internal i64 @"_ZZN7lambdas7LambdasEPcENK3$_0clEPvU17pass_object_size0"
// CHECK-NOT: call i64 @llvm.objectsize
// CHECK-DAG: define internal i64 @"_ZZN7lambdas7LambdasEPcENK3$_1clEPvU17pass_object_size0"
// CHECK-NOT: call i64 @llvm.objectsize
}
