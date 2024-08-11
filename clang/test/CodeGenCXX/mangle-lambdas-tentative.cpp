// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-linux -emit-llvm -o - %s -w | FileCheck %s

namespace declared_on_param {
  // CHECK-LABEL: define dso_local void @_ZN17declared_on_param1fEv(
  void f() {
    struct B {
      B(decltype([]{}) = {}) {}
    };
    B b;
  }
  // CHECK: call void @"_ZZN17declared_on_param1fEvEN1BC1ENS0_3$_0E"(
} // namespace declared_on_param
