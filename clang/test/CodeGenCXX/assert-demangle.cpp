// RUN: %clang_cc1 -emit-llvm -fno-demangling-failures -triple %itanium_abi_triple -o - %s | FileCheck %s

// CHECK: @_ZN6foobar3barEv
// CHECK: @_ZN6foobar1A3fooEi
namespace foobar {
struct A {
  void foo (int) {
  }
};

void bar() {
  A().foo(0);
}
} // namespace foobar
