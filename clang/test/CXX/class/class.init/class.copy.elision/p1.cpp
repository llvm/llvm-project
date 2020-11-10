// RUN: %clang_cc1 -std=c++20 -emit-llvm -triple x86_64-unknown-linux-gnu -o - %s | FileCheck %s
// RUN: %clang_cc1 -std=c++17 -emit-llvm -triple x86_64-unknown-linux-gnu -o - %s | FileCheck %s
// RUN: %clang_cc1 -std=c++14 -emit-llvm -triple x86_64-unknown-linux-gnu -o - %s | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -emit-llvm -triple x86_64-unknown-linux-gnu -o - %s | FileCheck %s

// - volatile object in return statement don't match the rule for using move
//   operation instead of copy operation. Thus should call the copy constructor
//   A(const volatile A &).
//
// - volatile object in return statement also don't match the rule for copy
//   elision. Thus the copy constructor A(const volatile A &) cannot be elided.
namespace test_volatile {
class A {
public:
  A() {}
  ~A() {}
  A(const volatile A &);
  A(volatile A &&);
};

A test() {
  volatile A a_copy;
  // CHECK: call void @_ZN13test_volatile1AC1ERVKS0_
  return a_copy;
}
} // namespace test_volatile
