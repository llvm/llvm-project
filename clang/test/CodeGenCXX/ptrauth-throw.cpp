// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fcxx-exceptions -emit-llvm %s -o - | FileCheck %s

class Foo {
 public:
  ~Foo() {
  }
};

void f() {
  throw Foo();
}

// CHECK: @_ZN3FooD1Ev.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @_ZN3FooD1Ev, i32 0, i64 0, i64 0 }, section "llvm.ptrauth", align 8

// CHECK: define void @_Z1fv()
// CHECK:  call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTI3Foo, ptr @_ZN3FooD1Ev.ptrauth)
