// REQUIRES: asserts
// RUN: %clang_cc1 -O0 -triple %itanium_abi_triple -debug-info-kind=limited -S -emit-llvm %s -o - | \
// RUN: FileCheck %s

// This test simply checks that the varargs thunk is created. The failing test
// case asserts.

struct Alpha {
  virtual void bravo(...);
};
struct Charlie {
  virtual ~Charlie() {}
};
struct CharlieImpl : Charlie, Alpha {
  void bravo(...) {}
} delta;

// CHECK: define {{.*}} void @_ZThn{{[48]}}_N11CharlieImpl5bravoEz(
