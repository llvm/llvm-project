// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

// A friend function template defined inside a class template, where the
// declaration found by the call and the one holding the definition have
// different template contexts. The definition used to be instantiated in the
// wrong context, so `foo<U>{}` stayed dependent and IR generation crashed.

template <typename T>
struct foo {
  template <typename U>
  friend foo<U> make_foo() { return foo<U>{}; }
};

template <typename U>
foo<U> make_foo();

void f() {
  make_foo<int>();
}

// CHECK-LABEL: define {{.*}}void @_Z1fv()
// CHECK:         call void @_Z8make_fooIiE3fooIT_Ev()
// CHECK-LABEL: define linkonce_odr void @_Z8make_fooIiE3fooIT_Ev()
// CHECK:         ret void
