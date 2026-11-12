// RUN: %clang_cc1 %s -triple=i686-pc-win32 -fms-extensions -stack-protector 2 -emit-llvm -o - | FileCheck %s

#pragma strict_gs_check(on)

extern "C" void test0() {
}
// CHECK: define {{.*}} @test0() #[[ATTR_TEST0:[0-9]*]] {


// CHECK: attributes #[[ATTR_TEST0]] = {{.*}} sspstrong

