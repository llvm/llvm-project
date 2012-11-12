// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - -fcxx-exceptions -fexceptions | FileCheck %s

void external();

void target() throw(int)
{
  // CHECK: invoke void @_Z8externalv()
  external();
}
// CHECK:      landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
// CHECK-NEXT:   filter [1 x i8*] [i8* bitcast (i8** @_ZTIi to i8*)]
// CHECK:      call void @__cxa_call_unexpected
