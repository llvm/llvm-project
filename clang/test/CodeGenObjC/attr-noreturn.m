// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-MRC
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-ARC

__attribute__((objc_root_class))
@interface Root
- (instancetype) init;
@end

@interface Base : Root
@end

@interface Middle : Base
+ (void) abort __attribute__((noreturn));
- (void) fail __attribute__((noreturn));
@end
  
@interface Derived : Middle
@end

// An arbitrary instance pointer may be null.
void testInstanceMethod(Derived *x) {
  [x fail];
}
// CHECK-LABEL: @testInstanceMethod
// CHECK: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*)*)(i8* {{.*}}, i8* {{.*}}){{$}}

// A direct call of a class method will normally never have a null receiver.
void testClassMethod(void) {
  [Derived abort];
}
// CHECK-LABEL: @testClassMethod
// CHECK: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*)*)(i8* {{.*}}, i8* {{.*}}) [[NORETURN:#[0-9]+]]

__attribute__((weak_import))
@interface WeakMiddle : Base
@end
  
@interface WeakDerived : WeakMiddle
+ (void) abort __attribute__((noreturn));
@end

// The class pointer of a weakly-imported class may be null.
void testWeakImport(void) {
  [WeakDerived abort];
}
// CHECK-LABEL: @testWeakImport
// CHECK: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*)*)(i8* {{.*}}, i8* {{.*}}){{$}}

@interface Derived (MyMethods)
@end

@implementation Derived (MyMethods)

// In general, self can be reassigned, so we can't make stronger assumptions.
// But ARC makes self const in an ordinary method.
// TODO: do the analysis to take advantage of the dominant case where
// self is not reassigned.
- (void) testSelfInstanceMethod {
  [self fail];
}
// CHECK-LABEL: [Derived(MyMethods) testSelfInstanceMethod]
// CHECK-MRC: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*)*)(i8* {{.*}}, i8* {{.*}}){{$}}
// CHECK-ARC: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*)*)(i8* {{.*}}, i8* {{.*}}) [[NORETURN]]

// The ARC rule doesn't apply in -init methods.
- (id) initWhileTestingSelfInstanceMethod {
  self = [super init];
  [self fail];
  return self;
}
// CHECK-LABEL: [Derived(MyMethods) initWhileTestingSelfInstanceMethod]
// CHECK: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*)*)(i8* {{.*}}, i8* {{.*}}){{$}}

// Same thing applies to class methods.
+ (void) testSelfClassMethod {
  [self abort];
}
// CHECK-LABEL: [Derived(MyMethods) testSelfClassMethod]
// CHECK-MRC: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*)*)(i8* {{.*}}, i8* {{.*}}){{$}}
// CHECK-ARC: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*)*)(i8* {{.*}}, i8* {{.*}}) [[NORETURN]]

// Super invocations may never be used with a null pointer; this is a
// constraint on user code when it isn't enforced by the ARC const-self
// rule.
- (void) testSuperInstanceMethod {
  [super fail];
}
// CHECK-LABEL: [Derived(MyMethods) testSuperInstanceMethod]
// CHECK: call void bitcast (i8* ([[SUPER_T:%.*]]*, i8*, ...)* @objc_msgSendSuper2 to void ([[SUPER_T]]*, i8*)*)([[SUPER_T]]* {{.*}}, i8* {{.*}}) [[NORETURN]]

+ (void) testSuperClassMethod {
  [super abort];
}
// CHECK-LABEL: [Derived(MyMethods) testSuperClassMethod]
// CHECK: call void bitcast (i8* ([[SUPER_T]]*, i8*, ...)* @objc_msgSendSuper2 to void ([[SUPER_T]]*, i8*)*)([[SUPER_T]]* {{.*}}, i8* {{.*}}) [[NORETURN]]
@end

// CHECK: attributes [[NORETURN]] = { noreturn }
  