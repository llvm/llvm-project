// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

struct Foo {
  Foo();
  Foo(const Foo&);
};

struct Bar {
  Bar();
  operator const Foo&() const;
};

void f(Foo);

// CHECK-LABEL: define{{.*}} void @_Z1g3Foo(%struct.Foo* noundef %foo)
void g(Foo foo) {
  // CHECK: call void @_ZN3BarC1Ev
  // CHECK: @_ZNK3BarcvRK3FooEv
  // CHECK: call void @_Z1f3Foo
  f(Bar());
  // CHECK: call void @_ZN3FooC1Ev
  // CHECK: call void @_Z1f3Foo
  f(Foo());
  // CHECK: call void @_ZN3FooC1ERKS_
  // CHECK: call void @_Z1f3Foo
  f(foo);
  // CHECK: ret
}

