// REQUIRES: x86-registered-target
// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -fmodules-cache-path=%t -emit-module -fmodule-name=irgen -triple x86_64-apple-darwin10 %S/Inputs/module.modulemap
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -fmodules-cache-path=%t -I %S/Inputs -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s
// FIXME: When we have a syntax for modules in C, use that.

@import irgen;

// CHECK: define{{.*}} void @triple_value
void triple_value(int *px) {
  *px = triple(*px);
}

// CHECK: define internal i32 @triple(i32
