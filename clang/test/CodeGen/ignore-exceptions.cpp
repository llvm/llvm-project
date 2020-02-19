// RUN: %clang_cc1 %s -triple powerpc64-linux -fexceptions -fcxx-exceptions -fignore-exceptions -emit-llvm -o - | FileCheck %s

struct A {
  ~A(){}
};

void f(void) {
// CHECK-NOT: personality i8* bitcast (i32 (...)* @__gcc_personality_v0 to i8*)
  A a;
  try {
    throw 1;
  } catch(...) {
  }
// CHECK:  %a = alloca %struct.A, align 1
// CHECK:  %exception = call i8* @__cxa_allocate_exception(i64 4) #1
// CHECK:  %0 = bitcast i8* %exception to i32*
// CHECK:  store i32 1, i32* %0, align 16
// CHECK:  call void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null) #2
// CHECK:  unreachable

// CHECK-NOT: invoke
// CHECK-NOT: landingpad
// CHECK-NOT: __cxa_begin_catch
// CHECK-NOT: __cxa_end_catch
}
