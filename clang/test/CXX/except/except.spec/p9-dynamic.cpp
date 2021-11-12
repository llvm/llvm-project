// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - -fcxx-exceptions -fexceptions | FileCheck %s --check-prefixes=CHECK,CHECK-PRE17
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -std=c++17 -Wno-dynamic-exception-spec -emit-llvm -o - -fcxx-exceptions -fexceptions | FileCheck %s --check-prefixes=CHECK,CHECK-17

void external();

// CHECK-LABEL: _Z6targetv(
// CHECK: invoke void @_Z8externalv()
// CHECK:      landingpad { i8*, i32 }
// CHECK-NEXT:   filter [1 x i8*] [i8* bitcast (i8** @_ZTIi to i8*)]
// CHECK:      call void @__cxa_call_unexpected
void target() throw(int)
{
  external();
}

// CHECK-LABEL: _Z7target2v(
// CHECK: invoke void @_Z8externalv()
// CHECK:            landingpad { i8*, i32 }
// CHECK-PRE17-NEXT:   filter [0 x i8*] zeroinitializer
// CHECK-17-NEXT:      catch i8* null
// CHECK-PRE17:      call void @__cxa_call_unexpected
// CHECK-17:         call void @__clang_call_terminate
void target2() throw()
{
  external();
}
