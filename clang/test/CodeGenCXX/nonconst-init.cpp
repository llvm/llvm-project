// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s %}

int a();
// CHECK: call noundef i32 @_Z1av()
struct x {int x, y : 10;} x = {1, a()};
