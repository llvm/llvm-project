// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsymbol-partition=foo -emit-llvm -o - %s | FileCheck %s

// CHECK: @gv = {{.*}}, partition "foo"
// CHECK: @_ZTV1S = {{.*}}, partition "foo"
// CHECK: @_ZTI1S = {{.*}}, partition "foo"
// CHECK: @_ZTS1S = {{.*}}, partition "foo"

// CHECK: @_Z5ifuncv = {{.*}}, partition "foo"

// CHECK: define {{.*}} @_ZN1S1fEv({{.*}} partition "foo"
// CHECK: define {{.*}} @f({{.*}} partition "foo"

struct S {
  virtual void f();
};

void S::f() {}

int gv;
extern "C" void *f() { return 0; }
void ifunc() __attribute__((ifunc("f")));
