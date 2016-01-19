// RUN: %clang_cc1 -analyze -analyzer-checker=core,nullability -verify %s

#define nil 0

@protocol NSObject
+ (id)alloc;
- (id)init;
@end

__attribute__((objc_root_class))
@interface
NSObject<NSObject>
@end

@interface TestObject : NSObject
@end

TestObject * _Nonnull returnsNilObjCInstanceIndirectly() {
  TestObject *local = 0;
  return local; // expected-warning {{Null is returned from a function that is expected to return a non-null value}}
}

TestObject * _Nonnull returnsNilObjCInstanceIndirectlyWithSupressingCast() {
  TestObject *local = 0;
  return (TestObject * _Nonnull)local; // no-warning
}

TestObject * _Nonnull returnsNilObjCInstanceDirectly() {
  // The first warning is from Sema. The second is from the static analyzer.
  return nil; // expected-warning {{null returned from function that requires a non-null return value}}
              // expected-warning@-1 {{Null is returned from a function that is expected to return a non-null value}}
}

TestObject * _Nonnull returnsNilObjCInstanceDirectlyWithSuppressingCast() {
  return (TestObject * _Nonnull)nil; // no-warning
}

void testObjCNonARCNoInitialization(TestObject * _Nonnull p) {
  TestObject * _Nonnull implicitlyZeroInitialized; // no-warning
  implicitlyZeroInitialized = p;
}

void testObjCNonARCExplicitZeroInitialization() {
  TestObject * _Nonnull explicitlyZeroInitialized = nil; // expected-warning {{Null is assigned to a pointer which is expected to have non-null value}}
}
