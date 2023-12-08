// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin10 -emit-llvm -fcxx-exceptions -fexceptions -fobjc-exceptions -o - %s | FileCheck %s

@interface OCType @end
void opaque();

namespace test0 {

  // CHECK-LABEL: define{{.*}} void @_ZN5test03fooEv
  // CHECK-SAME:  personality ptr @__objc_personality_v0
  void foo() {
    try {
      // CHECK: invoke void @_Z6opaquev
      opaque();
    } catch (OCType *T) {
      // CHECK:      landingpad { ptr, i32 }
      // CHECK-NEXT:   catch ptr @"OBJC_EHTYPE_$_OCType"
    }
  }
}

@interface NSException
  + new;
@end
namespace test1 {

  void bar() {
    @try {
      throw [NSException new];
    } @catch (id i) {
    }
  }
// CHECK: invoke void @objc_exception_throw(ptr [[CALL:%.*]]) [[NR:#[0-9]+]]
// CHECK:          to label [[INVOKECONT1:%.*]] unwind label [[LPAD:%.*]]
}

// CHECK: attributes [[NR]] = { noreturn }
