// RUN: %clang_cc1 -std=c++2a -triple x86_64-apple-macosx10.7.0 -emit-llvm -o - %s -w | FileCheck %s

// CHECK: @__const._Z1fv.arr = private unnamed_addr constant [3 x i32] [i32 1, i32 2, i32 3], align 4

void f() {
  // CHECK: %[[ARR:.*]] = alloca [3 x i32], align 4
  // CHECK: call void @llvm.memcpy{{.*}}({{.*}} @__const._Z1fv.arr
  for (int arr[3] = {1, 2, 3}; int a : arr)
    ;
}
