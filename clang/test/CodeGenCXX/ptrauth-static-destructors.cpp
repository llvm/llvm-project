// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -emit-llvm -std=c++11 %s -o - \
// RUN:  | FileCheck %s --check-prefix=CXAATEXIT

// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -emit-llvm -std=c++11 %s -o - \
// RUN:    -fno-use-cxa-atexit \
// RUN:  | FileCheck %s --check-prefix=ATEXIT

class Foo {
 public:
  ~Foo() {
  }
};

Foo global;

// CXAATEXIT: @_ZN3FooD1Ev.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (%class.Foo* (%class.Foo*)* @_ZN3FooD1Ev to i8*), i32 0, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CXAATEXIT: define internal void @__cxx_global_var_init()
// CXAATEXIT:   call i32 @__cxa_atexit(void (i8*)* bitcast ({ i8*, i32, i64, i64 }* @_ZN3FooD1Ev.ptrauth to void (i8*)*), i8* getelementptr inbounds (%class.Foo, %class.Foo* @global, i32 0, i32 0), i8* @__dso_handle)


// ATEXIT: @__dtor_global.ptrauth = private constant { i8*, i32, i64, i64 } { i8* bitcast (void ()* @__dtor_global to i8*), i32 0, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// ATEXIT: define internal void @__cxx_global_var_init()
// ATEXIT:   %{{.*}} = call i32 @atexit(void ()* bitcast ({ i8*, i32, i64, i64 }* @__dtor_global.ptrauth to void ()*))

// ATEXIT: define internal void @__dtor_global() {{.*}} section "__TEXT,__StaticInit,regular,pure_instructions" {
// ATEXIT:   %{{.*}} = call %class.Foo* @_ZN3FooD1Ev(%class.Foo* @global)
