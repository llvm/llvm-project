// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 %s -emit-llvm -o - | FileCheck %s

namespace PR50787 {
// This code would previously cause a crash.
extern int x_;
consteval auto& X() { return x_; }
constexpr auto& x1 = X();
auto x2 = X();

// CHECK: @_ZN7PR507872x_E = external global i32, align 4
// CHECK-NEXT: @_ZN7PR507872x1E = constant i32* @_ZN7PR507872x_E, align 8
// CHECK-NEXT: @_ZN7PR507872x2E = global i32* @_ZN7PR507872x_E, align 4
}

