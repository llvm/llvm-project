// RUN: %clang_cc1 -triple=x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s

// CHECK: [[FOO:%.+]] = type { i32 }
struct foo {
  int val;
};

template <typename T> struct bar : T {
};

struct baz : bar<foo> {
  // CHECK-LABEL: define{{.*}} i32 @_ZN3baz3getEv
  // CHECK: {{%.+}} = getelementptr inbounds [[FOO]], ptr {{%.+}}, i32 0, i32 0
  int get() {
    return val;
  }
};

int qux() {
  auto f = baz{};
  return f.get();
}
