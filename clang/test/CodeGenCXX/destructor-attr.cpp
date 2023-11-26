// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -o - %s | FileCheck %s

// CHECK: @llvm.global_dtors
// CHECK-SAME: i32 101, ptr @_Z18template_dependentILi101EEvv
// CHECK-SAME: i32 108, ptr @_Z18template_dependentILi108EEvv

// PR6521
void bar();
struct Foo {
  // CHECK-LABEL: define linkonce_odr {{.*}}void @_ZN3Foo3fooEv
  static void foo() __attribute__((constructor)) {
    bar();
  }
};

template <int P>
[[gnu::destructor(P)]] void template_dependent() {}

template void template_dependent<101>();
template void template_dependent<100 + 8>();
