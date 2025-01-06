// RUN: %clang_cc1                                                -triple arm64-apple-ios   -fptrauth-calls -fcxx-exceptions -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -fptrauth-function-pointer-type-discrimination -triple arm64-apple-ios   -fptrauth-calls -fcxx-exceptions -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECKDISC

// RUN: %clang_cc1                                                -triple aarch64-linux-gnu -fptrauth-calls -fcxx-exceptions -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -fptrauth-function-pointer-type-discrimination -triple aarch64-linux-gnu -fptrauth-calls -fcxx-exceptions -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECKDISC

class Foo {
 public:
  ~Foo() {
  }
};

// CHECK-LABEL: define{{.*}} void @_Z1fv()
// CHECK:  call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTI3Foo, ptr ptrauth (ptr @_ZN3FooD1Ev, i32 0))

// CHECKDISC-LABEL: define{{.*}} void @_Z1fv()
// CHECKDISC:  call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTI3Foo, ptr ptrauth (ptr @_ZN3FooD1Ev, i32 0, i64 10942))

void f() {
  throw Foo();
}

// __cxa_throw is defined to take its destructor as "void (*)(void *)" in the ABI.
// CHECK-LABEL: define{{.*}} void @__cxa_throw({{.*}})
// CHECK: call void {{%.*}}(ptr noundef {{%.*}}) [ "ptrauth"(i32 0, i64 0) ]

// CHECKDISC-LABEL: define{{.*}} void @__cxa_throw({{.*}})
// CHECKDISC: call void {{%.*}}(ptr noundef {{%.*}}) [ "ptrauth"(i32 0, i64 10942) ]

extern "C" void __cxa_throw(void *exception, void *, void (*dtor)(void *)) {
  dtor(exception);
}
