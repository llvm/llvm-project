// RUN: %clang_cc1 -fms-extensions -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

struct A {
  ~A() {}
};

extern "C" int f() {
// CHECK: define i32 @f()
// CHECK-NOT: call void @_ZN1AD1Ev
// CHECK: ret i32 0
  return __noop(A());
};

extern "C" int g() {
  return __noop;
// CHECK: define i32 @g()
// CHECK: ret i32 0
}

extern "C" int h() {
  return (__noop);
// CHECK: define i32 @h()
// CHECK: ret i32 0
}

extern "C" int i() {
  return __noop + 1;
// CHECK: define i32 @i()
// CHECK: ret i32 1
}
