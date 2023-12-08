// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm -triple %itanium_abi_triple -o - %s | FileCheck %s

module;

#include "Inputs/module-extern-C.h"

export module x;

// CHECK: void @foo()
extern "C" void foo() {
  return;
}

extern "C" {
// CHECK: void @bar()
void bar() {
  return;
}
// CHECK: i32 @baz()
int baz() {
  return 3;
}
// CHECK: double @double_func()
double double_func() {
  return 5.0;
}
}
