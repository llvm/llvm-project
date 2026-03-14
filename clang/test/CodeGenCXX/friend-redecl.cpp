// RUN: %clang_cc1 -triple i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s
// PR8864

struct Foo {
  friend bool TryFoo(Foo *f2) { return TryFoo(0, f2); }

// CHECK: define{{.*}}Z6TryFooP3Foo
// CHECK-NOT: ret
// CHECK: call{{.*}}Z6TryFooiP3Foo
// CHECK: ret

  friend bool TryFoo(int, Foo *f3);
};
bool TryFoo(Foo *f5);
int main(void) {
  Foo f;
  TryFoo(&f);
}
