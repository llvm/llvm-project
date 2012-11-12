// REQUIRES: ppc64-registered-target
// RUN: %clang_cc1 -O0 -triple powerpc64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

void f1(int x) { return; }
// CHECK: define void @f1(i32 signext %x) nounwind 

void f2(unsigned int x) { return; }
// CHECK: define void @f2(i32 zeroext %x) nounwind 

int f3(void) { return 0; }
// CHECK: define signext i32 @f3() nounwind

unsigned int f4(void) { return 0; }
// CHECK: define zeroext i32 @f4() nounwind

