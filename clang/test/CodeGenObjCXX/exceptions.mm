// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fcxx-exceptions -fexceptions -o - %s | FileCheck %s

@interface OCType @end
void opaque();

namespace test0 {

  // CHECK: define void @_ZN5test03fooEv
  void foo() {
    try {
      // CHECK: invoke void @_Z6opaquev
      opaque();
    } catch (OCType *T) {
      // CHECK:      landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__objc_personality_v0 to i8*)
      // CHECK-NEXT:   catch %struct._objc_typeinfo* @"OBJC_EHTYPE_$_OCType"
    }
  }
}
