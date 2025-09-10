// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -o - %s | FileCheck %s

// CHECK: @llvm.global_ctors
// CHECK-SAME: i32 65535, ptr @_ZN3Foo3fooEv
// CHECK-SAME: i32 101, ptr @_Z22template_dependent_cxxILi101EEvv
// CHECK-SAME: i32 102, ptr @_Z22template_dependent_gnuILi102EEvv
// CHECK-SAME: i32 103, ptr @_Z1fv
// CHECK-SAME: i32 104, ptr @_Z23template_dependent_nttpIiLi104EEvv

// PR6521
void bar();
struct Foo {
  // CHECK-LABEL: define linkonce_odr {{.*}}void @_ZN3Foo3fooEv
  static void foo() __attribute__((constructor)) {
    bar();
  }
};


template <int P>
[[gnu::constructor(P)]] void template_dependent_cxx() {}
template <int P>
__attribute__((constructor(P))) void template_dependent_gnu() {}
template <typename T, int P = sizeof(T) * 26>
[[gnu::constructor(P)]] void template_dependent_nttp() {}

template void template_dependent_cxx<101>();
template void template_dependent_gnu<102>();
[[gnu::constructor(103)]] void f() {}
template void template_dependent_nttp<int>();
