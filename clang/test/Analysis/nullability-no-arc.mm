// RUN: %clang_cc1 -analyze -analyzer-checker=core,nullability -verify %s

#define nil 0

@protocol NSObject
+ (id)alloc;
- (id)init;
- (instancetype)autorelease;
- (void)release;
@end

__attribute__((objc_root_class))
@interface
NSObject<NSObject>
@end

@interface TestObject : NSObject
@end

TestObject *_Nonnull returnsNilObjCInstanceIndirectly() {
  TestObject *local = nil;
  return local; // expected-warning {{nil returned from a function that is expected to return a non-null value}}
}

TestObject * _Nonnull returnsNilObjCInstanceIndirectlyWithSupressingCast() {
  TestObject *local = nil;
  return (TestObject * _Nonnull)local; // no-warning
}

TestObject * _Nonnull returnsNilObjCInstanceDirectly() {
  // The first warning is from Sema. The second is from the static analyzer.
  return nil; // expected-warning {{null returned from function that requires a non-null return value}}
              // expected-warning@-1 {{nil returned from a function that is expected to return a non-null value}}
}

TestObject * _Nonnull returnsNilObjCInstanceDirectlyWithSuppressingCast() {
  return (TestObject * _Nonnull)nil; // no-warning
}

void testObjCNonARCNoInitialization(TestObject * _Nonnull p) {
  TestObject * _Nonnull implicitlyZeroInitialized; // no-warning
  implicitlyZeroInitialized = p;
}

void testObjCNonARCExplicitZeroInitialization() {
  TestObject * _Nonnull explicitlyZeroInitialized = nil; // expected-warning {{nil assigned to a pointer which is expected to have non-null value}}
}

@interface ClassWithInitializers : NSObject
@end

@implementation ClassWithInitializers
- (instancetype _Nonnull)initWithNonnullReturnAndSelfCheckingIdiom {
  // This defensive check is a common-enough idiom that we don't want
  // to issue a diagnostic for it.
  if (self = [super init]) {
  }

  return self; // no-warning
}

- (instancetype _Nonnull)initWithNonnullReturnAndNilReturnViaLocal {
  self = [super init];
  // This leaks, but we're not checking for that here.

  ClassWithInitializers *other = nil;
  // False negative. Once we have more subtle suppression of defensive checks in
  // initializers we should warn here.
  return other;
}

- (instancetype _Nonnull)initWithPreconditionViolation:(int)p {
  self = [super init];
  if (p < 0) {
    [self release];
    return (ClassWithInitializers * _Nonnull)nil;
  }
  return self;
}

+ (instancetype _Nonnull)factoryCallingInitWithNonnullReturnAndSelfCheckingIdiom {
  return [[[self alloc] initWithNonnullReturnAndSelfCheckingIdiom] autorelease]; // no-warning
}

+ (instancetype _Nonnull)factoryCallingInitWithNonnullReturnAndNilReturnViaLocal {
  return [[[self alloc] initWithNonnullReturnAndNilReturnViaLocal] autorelease]; // no-warning
}

+ (instancetype _Nonnull)initWithPreconditionViolation:(int) p {
  return [[[self alloc] initWithPreconditionViolation:p] autorelease]; // no-warning
}

- (TestObject * _Nonnull) returnsNil {
  return (TestObject * _Nonnull)nil;
}
- (TestObject * _Nonnull) inlineOfReturnsNilObjCInstanceDirectlyWithSuppressingCast {
  TestObject *o = [self returnsNil];
  return o;
}
@end
