// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

void f1(int x) { return; }
// CHECK: define void @f1(i32 signext %x) [[NUW:#[0-9]+]]

void f2(unsigned int x) { return; }
// CHECK: define void @f2(i32 zeroext %x) [[NUW]]

int f3(void) { return 0; }
// CHECK: define signext i32 @f3() [[NUW]]

unsigned int f4(void) { return 0; }
// CHECK: define zeroext i32 @f4() [[NUW]]

// CHECK: attributes [[NUW]] = { noinline nounwind{{.*}} }
