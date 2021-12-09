// RUN: %clang_cc1 -std=c++20 -emit-llvm -triple %itanium_abi_triple -o - %s | FileCheck %s

module;

#include "Inputs/module-extern-C.h"

export module x;

// CHECK: define dso_local void @foo()
extern "C" void foo() {
  return;
}

extern "C" {
// CHECK: define dso_local void @bar()
void bar() {
  return;
}
// CHECK: define dso_local i32 @baz()
int baz() {
  return 3;
}
// CHECK: define dso_local double @double_func()
double double_func() {
  return 5.0;
}
}
