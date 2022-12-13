// RUN: %clang_cc1 %std_cxx98-14 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - -fcxx-exceptions -fexceptions | FileCheck %s --check-prefixes=CHECK,CHECK-PRE17
// RUN: %clang_cc1 %std_cxx17- %s -triple=x86_64-apple-darwin10 -Wno-dynamic-exception-spec -emit-llvm -o - -fcxx-exceptions -fexceptions | FileCheck %s --check-prefixes=CHECK,CHECK-17

void external();

// CHECK-LABEL: _Z6targetv(
// CHECK: invoke void @_Z8externalv()
// CHECK:      landingpad { ptr, i32 }
// CHECK-NEXT:   filter [1 x ptr] [ptr @_ZTIi]
// CHECK:      call void @__cxa_call_unexpected
void target() throw(int)
{
  external();
}

// CHECK-LABEL: _Z7target2v(
// CHECK: invoke void @_Z8externalv()
// CHECK:            landingpad { ptr, i32 }
// CHECK-PRE17-NEXT:   filter [0 x ptr] zeroinitializer
// CHECK-17-NEXT:      catch ptr null
// CHECK-PRE17:      call void @__cxa_call_unexpected
// CHECK-17:         call void @__clang_call_terminate
void target2() throw()
{
  external();
}
