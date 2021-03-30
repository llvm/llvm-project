// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fcxx-exceptions -emit-llvm %s -o - | FileCheck %s

class Foo {
 public:
  ~Foo() {
  }
};

void f() {
  throw Foo();
}

// CHECK: @_ZN3FooD1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%class.Foo* (%class.Foo*)* @_ZN3FooD1Ev to i8*), i32 0, i64 0, i64 0 }, section "llvm.ptrauth", align 8

// CHECK: define void @_Z1fv()
// CHECK:  call void @__cxa_throw(i8* %{{.*}}, i8* bitcast ({ i8*, i8* }* @_ZTI3Foo to i8*), i8* bitcast ({ i8*, i32, i64, i64 }* @_ZN3FooD1Ev.ptrauth to i8*))
