// RUN: %clang_cc1 -no-opaque-pointers -x objective-c++ -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-10.7 -emit-llvm -o - %s | FileCheck %s

// Properly instantiate a non-dependent message expression which
// requires a contextual conversion to ObjC pointer type.
// <rdar://13305374>
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
  // CHECK:      [[T0:%.*]] = call noundef [[TEST0:%.*]]* @_ZN5test01AcvP5Test0Ev(
  // CHECK-NEXT: [[T2:%.*]] = bitcast [[TEST0]]* [[T0]] to i8*
  // CHECK-NEXT: [[T1:%.*]] = load i8*, i8**
  // CHECK-NEXT: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*)*)(i8* noundef [[T2]], i8* noundef [[T1]])
  // CHECK-NEXT: ret void
}
