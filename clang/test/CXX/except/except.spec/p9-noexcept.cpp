// RUN: %clang_cc1 %s -std=c++11 -triple=x86_64-apple-darwin10 -emit-llvm -o - -fcxx-exceptions -fexceptions | FileCheck %s

void external();

void target() noexcept
{
  // CHECK: invoke void @_Z8externalv()
  external();
}
// CHECK: landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
// CHECK-NEXT:  catch i8* null
// CHECK-NEXT: call void @_ZSt9terminatev() noreturn nounwind
// CHECK-NEXT: unreachable

void reverse() noexcept(false)
{
  // CHECK: call void @_Z8externalv()
  external();
}
