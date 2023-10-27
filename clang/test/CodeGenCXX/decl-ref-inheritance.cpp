// RUN: %clang_cc1 -triple=x86_64-unknown-linux -emit-llvm %s -o - | FileCheck \
// RUN: -check-prefix=CHECK-1 %s
// RUN: %clang_cc1 -triple=x86_64-unknown-linux -emit-llvm %s -o - | FileCheck \
// RUN: -check-prefix=CHECK-2 %s
// RUN: %clang_cc1 -triple=x86_64-unknown-linux -emit-llvm %s -o - | FileCheck \
// RUN: -check-prefix=CHECK-3 %s

// CHECK-1: [[FOO:%.+]] = type { float }
struct foo {
  float val;
};

template <typename T> struct bar : T {
};

struct baz : bar<foo> {
  // CHECK-1: define{{.*}} float @_ZN3baz3getEv
  // CHECK-1: {{%.+}} = getelementptr inbounds [[FOO]], ptr {{%.+}}, i32 0, i32 0
  float get() {
    return val;
  }
};

int qux() {
  auto f = baz{};
  return f.get();
}

// CHECK-2: [[F:%.+]] = type { ptr }
struct f {
  void *g;
};

template <typename j> struct k : j {
  // CHECK-2: define{{.*}} void @_ZN1kI1fE1lEv
  // CHECK-2: {{%.+}} = getelementptr inbounds [[F]], ptr {{%.+}}, i32 0, i32 0
  virtual void l(){ (void)f::g; }
};

k<f> q;

// CHECK-3: [[BASE:%.+]] = type { i32 }
class Base {
protected:
  int member;
};

template <typename Parent>
struct Subclass : public Parent {
  // CHECK-3: define{{.*}} i32 @_ZN8SubclassI4BaseE4funcEv
  // CHECK-3: {{%.+}} = getelementptr inbounds [[BASE]], ptr {{%.+}}, i32 0, i32 0
  int func() { return Base::member; }
};

using Impl = Subclass<Base>;

int use() {
  Impl i;
  return i.func();
}
