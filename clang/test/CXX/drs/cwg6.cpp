// RUN: %clang_cc1 -std=c++98 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++11 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++14 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++17 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++20 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++23 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++2c %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | FileCheck %s --check-prefixes CHECK

namespace cwg6 { // cwg6: yes

struct A {
  A();
  A(const A&);
  ~A();
};

inline void f(A a) {}

// CHECK-LABEL: define {{.*}} @_ZN4cwg64callEv
void call() {
  A a;
  // We copy the parameter here, even though object is not mutated by f and
  // otherwise satisfies the criteria for the proposed CWG6 optimization.
  // CHECK: {{call|invoke}} {{.*}} @_ZN4cwg61AC1ERKS0_
  // CHECK: {{call|invoke}} {{.*}} @_ZN4cwg61fE
  f(a);
  // CHECK: {{call|invoke}} {{.*}} @_ZN4cwg61AD1Ev
  // CHECK: {{call|invoke}} {{.*}} @_ZN4cwg61AD1Ev
}

} // namespace cwg6
