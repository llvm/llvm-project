// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-10.7 -emit-llvm -o - %s | FileCheck %s

// Properly instantiate a non-dependent message expression which
// requires a contextual conversion to ObjC pointer type.
@interface Test0
- (void) foo;
@end
namespace test0 {
  struct A {
    operator Test0*();
  };
  template <class T> void foo() {
    A a;
    [a foo];
  }
  template void foo<int>();
  // CHECK-LABEL:    define weak_odr void @_ZN5test03fooIiEEvv()
  // CHECK:      [[T0:%.*]] = call noundef ptr @_ZN5test01AcvP5Test0Ev(
  // CHECK-NEXT: [[T1:%.*]] = load ptr, ptr
  // CHECK-NEXT: call void @objc_msgSend(ptr noundef [[T0]], ptr noundef [[T1]])
  // CHECK-NEXT: ret void
}
