// RUN: %clang_cc1 -fmodules-ts %s -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s
export module FOO;
namespace Outer {
class Y;
class Inner {
  class X;
  void Fn (X &, Y &); // #2
};
// CHECK-DAG: void @_ZN5OuterW3FOO5Inner2FnERNS1_1XERNS_S0_1YE(
void Inner::Fn (X &, Y &) {}
}

