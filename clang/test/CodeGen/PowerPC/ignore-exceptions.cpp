// RUN: %clang_cc1 %s -triple powerpc64-linux -fexceptions -fcxx-exceptions -fignore-exceptions -emit-llvm -o - | FileCheck %s

struct A {
  ~A(){}
};

void f(void) {
// CHECK-NOT: personality ptr @__gcc_personality_v0
  A a;
  try {
    throw 1;
  } catch(...) {
  }
// CHECK:  %a = alloca %struct.A, align 1
// CHECK:  %exception = call ptr @__cxa_allocate_exception(i64 4) #1
// CHECK:  store i32 1, ptr %exception, align 16
// CHECK:  call void @__cxa_throw(ptr %exception, ptr @_ZTIi, ptr null) #2
// CHECK:  unreachable

// CHECK-NOT: invoke
// CHECK-NOT: landingpad
// CHECK-NOT: __cxa_begin_catch
// CHECK-NOT: __cxa_end_catch
}
