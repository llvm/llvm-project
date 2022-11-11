// RUN: %clang_cc1 -std=c++14 -verify -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

template <class F> void parallel_loop(F &&f) { f(0); }

//CHECK-LABEL: @main
int main() {
// CHECK: [[X_ADDR:%.+]] = alloca i32,
  int x;
// CHECK: getelementptr inbounds
// CHECK: store ptr [[X_ADDR]], ptr %
// CHECK: call
  parallel_loop([&](auto y) {
#pragma clang __debug captured
    {
      x = y;
    };
  });
}
