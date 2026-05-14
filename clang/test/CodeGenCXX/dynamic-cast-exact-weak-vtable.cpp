// RUN: %clangxx --target=x86_64-unknown-linux-gnu -O1 -fPIC -emit-llvm -S -std=c++11 -o - %s | FileCheck %s --check-prefixes=CHECK,INEXACT

struct Foo {
  virtual ~Foo();
};

// Header-defined final classes have a weak_odr/linkonce_odr vtable. In the
// ThinLTO shared-library reproducer for issue #71196, using that vtable as a
// unique identity for exact dynamic_cast is wrong because another linkage unit
// can end up with a different local copy.
struct Bar final : Foo {};

Bar *make() { return new Bar(); }

// CHECK: @_ZTV3Bar = linkonce_odr{{.*}}constant

// CHECK-LABEL: define {{.*}} @_Z4castP3Foo
Bar *cast(Foo *f) {
  // INEXACT: call {{.*}} @__dynamic_cast(ptr nonnull %[[ARG:.*]], ptr nonnull @_ZTI3Foo, ptr nonnull @_ZTI3Bar, i64 0)
  return dynamic_cast<Bar *>(f);
}
