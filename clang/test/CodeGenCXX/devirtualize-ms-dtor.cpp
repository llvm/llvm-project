// RUN: %clang_cc1 -std=c++11 -triple x86_64-windows-msvc %s -emit-llvm -o - | FileCheck %s

// If we de-virtualize ~Foo, we still need to call ??1Foo, not ??_DFoo.

struct Base {
  virtual ~Base();
};
struct Foo final : Base {
};
void f(Foo *p) {
  p->~Foo();
}

// CHECK-LABEL: define{{.*}} void @"?f@@YAXPEAUFoo@@@Z"(ptr noundef %p)
// CHECK: call void @"??1Foo@@UEAA@XZ"
// CHECK: ret void
