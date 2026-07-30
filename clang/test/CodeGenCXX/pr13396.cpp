// RUN: %clang_cc1 -triple i686-pc-linux-gnu %s -emit-llvm -o - | FileCheck %s
struct foo {
  template<typename T>
  __attribute__ ((regparm (3))) foo(T x) {}
  __attribute__ ((regparm (3))) foo();
  __attribute__ ((regparm (3))) ~foo();
};

foo::foo() {
  // CHECK-LABEL: define{{.*}} void @_ZN3fooC2Ev(ptr inreg noundef nonnull align 1 dereferenceable(1) %this)
  // CHECK-LABEL: define{{.*}} void @_ZN3fooC1Ev(ptr inreg noundef nonnull align 1 dereferenceable(1) %this)
}

foo::~foo() {
  // CHECK-LABEL: define{{.*}} void @_ZN3fooD2Ev(ptr inreg noundef nonnull align 1 dereferenceable(1) %this)
  // CHECK-LABEL: define{{.*}} void @_ZN3fooD1Ev(ptr inreg noundef nonnull align 1 dereferenceable(1) %this)
}

void dummy() {
  // FIXME: how can we explicitly instantiate a template constructor? Gcc and
  // older clangs accept:
  // template foo::foo(int x);
  foo x(10);
  // CHECK-LABEL: define linkonce_odr void @_ZN3fooC1IiEET_(ptr inreg noundef nonnull align 1 dereferenceable(1) %this, i32 inreg noundef %x)
  // CHECK-LABEL: define linkonce_odr void @_ZN3fooC2IiEET_(ptr inreg noundef nonnull align 1 dereferenceable(1) %this, i32 inreg noundef %x)
}
