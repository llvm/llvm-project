// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -x objective-c -fmodule-cache-path %t -emit-module -fmodule-name=irgen -triple x86_64-apple-darwin10 %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -x objective-c -fmodule-cache-path %t -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s
// FIXME: When we have a syntax for modules in C, use that.

@__experimental_modules_import irgen;

// CHECK: define void @triple_value
void triple_value(int *px) {
  *px = triple(*px);
}

// CHECK: define internal i32 @triple(i32
