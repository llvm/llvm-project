// RUN: %clang_cc1 -no-opaque-pointers %s -fblocks -triple x86_64-apple-darwin -std=c++11 -emit-llvm -o - | FileCheck %s
// REQUIRES: x86-registered-target

// CHECK: @b ={{.*}} global i32 0,

// CHECK: define {{.*}}void @{{.*}}test{{.*}}_block_invoke(
// CHECK: store i32 2, i32* @b,
// CHECK: ret void

int b;

void test() {
  int &a = b;
  ^{ a = 2; };
}
