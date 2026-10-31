// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-llvm %s -o - | FileCheck %s
int call(int (*fp)(int)) { return fp(1); }
// CHECK: call i32 %{{.*}}(i32 noundef 1)
