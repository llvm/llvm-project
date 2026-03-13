// RUN: %clang_cc1 -std=c++20 %s -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s

template <template <class> class S>
void create_unique()
  requires (S{0}, true) {}

template <class Fn> struct A {
  constexpr A(Fn) {};
};

template void create_unique<A>();
// CHECK: @_Z13create_uniqueI1AEvvQcmtlT_Li0EELb1E(
