// RUN: %clangxx --target=x86_64-unknown-linux-gnu -O1 -fPIC -emit-llvm -S -std=c++11 -o - %s | FileCheck %s --check-prefixes=CHECK

struct Foo {
  virtual ~Foo();
};

struct Bar final : Foo {};

Bar *make() { return new Bar(); }

// CHECK: @_ZTV3Bar = linkonce_odr{{.*}}constant

// CHECK-LABEL: define {{.*}} @_Z4castP3Foo
Bar *cast(Foo *f) {
  // CHECK: call {{.*}} @__dynamic_cast(ptr nonnull %[[ARG:.*]], ptr nonnull @_ZTI3Foo, ptr nonnull @_ZTI3Bar, i64 0)
  return dynamic_cast<Bar *>(f);
}
