// RUN: %clang_cc1 -analyze -analyzer-checker=core,osx.cocoa.SuperDealloc,debug.ExprInspection -analyzer-output=text -verify %s

void clang_analyzer_warnIfReached();

#define nil ((id)0)

typedef unsigned long NSUInteger;
@protocol NSObject
- (instancetype)retain;
- (oneway void)release;
@end

@interface NSObject <NSObject> { }
- (void)dealloc;
- (instancetype)init;
@end

typedef struct objc_selector *SEL;

//===------------------------------------------------------------------------===
//  <rdar://problem/6953275>
//  Check that 'self' is not referenced after calling '[super dealloc]'.

@interface SuperDeallocThenReleaseIvarClass : NSObject {
  NSObject *_ivar;
}
@end

@implementation SuperDeallocThenReleaseIvarClass
- (instancetype)initWithIvar:(NSObject *)ivar {
  self = [super init];
  if (!self)
    return nil;
  _ivar = [ivar retain];
  return self;
}
- (void)dealloc {
  [super dealloc]; // expected-note {{[super dealloc] called here}}
  [_ivar release]; // expected-warning {{Use of instance variable '_ivar' after 'self' has been deallocated}}
  // expected-note@-1 {{Use of instance variable '_ivar' after 'self' has been deallocated}}
}
@end

@interface SuperDeallocThenAssignNilToIvarClass : NSObject {
  NSObject *_delegate;
}
@end

@implementation SuperDeallocThenAssignNilToIvarClass
- (instancetype)initWithDelegate:(NSObject *)delegate {
  self = [super init];
  if (!self)
    return nil;
  _delegate = delegate;
  return self;
}
- (void)dealloc {
  [super dealloc]; // expected-note {{[super dealloc] called here}}
  _delegate = nil; // expected-warning {{Use of instance variable '_delegate' after 'self' has been deallocated}}
      // expected-note@-1 {{Use of instance variable '_delegate' after 'self' has been deallocated}}
}
@end


struct SomeStruct {
  int f;
};

@interface SuperDeallocThenAssignIvarField : NSObject {
  struct SomeStruct _s;
}
@end

@implementation SuperDeallocThenAssignIvarField
- (void)dealloc {
  [super dealloc]; // expected-note {{[super dealloc] called here}}
  _s.f = 7; // expected-warning {{Use of instance variable '_s' after 'self' has been deallocated}}
      // expected-note@-1 {{Use of instance variable '_s' after 'self' has been deallocated}}
}
@end

@interface OtherClassWithIvar {
@public
  int _otherIvar;
}
@end;

@interface SuperDeallocThenAssignIvarIvar : NSObject {
  OtherClassWithIvar *_ivar;
}
@end

@implementation SuperDeallocThenAssignIvarIvar
- (void)dealloc {
  [super dealloc]; // expected-note {{[super dealloc] called here}}
  _ivar->_otherIvar = 7; // expected-warning {{Use of instance variable '_ivar' after 'self' has been deallocated}}
      // expected-note@-1 {{Use of instance variable '_ivar' after 'self' has been deallocated}}
}
@end

@interface SuperDeallocThenAssignSelfIvar : NSObject {
  NSObject *_ivar;
}
@end

@implementation SuperDeallocThenAssignSelfIvar
- (void)dealloc {
  [super dealloc]; // expected-note {{[super dealloc] called here}}
  self->_ivar = nil; // expected-warning {{Use of instance variable '_ivar' after 'self' has been deallocated}}
      // expected-note@-1 {{Use of instance variable '_ivar' after 'self' has been deallocated}}
}
@end

@interface SuperDeallocThenReleasePropertyClass : NSObject { }
@property (retain) NSObject *ivar;
@end

@implementation SuperDeallocThenReleasePropertyClass
- (instancetype)initWithProperty:(NSObject *)ivar {
  self = [super init];
  if (!self)
    return nil;
  self.ivar = ivar;
  return self;
}
- (void)dealloc {
  [super dealloc]; // expected-note {{[super dealloc] called here}}
  self.ivar = nil; // expected-warning {{use of 'self' after it has been deallocated}}
      // expected-note@-1 {{use of 'self' after it has been deallocated}}
}
@end

@interface SuperDeallocThenAssignNilToPropertyClass : NSObject { }
@property (assign) NSObject *delegate;
@end

@implementation SuperDeallocThenAssignNilToPropertyClass
- (instancetype)initWithDelegate:(NSObject *)delegate {
  self = [super init];
  if (!self)
    return nil;
  self.delegate = delegate;
  return self;
}
- (void)dealloc {
  [super dealloc]; // expected-note {{[super dealloc] called here}}
  self.delegate = nil; // expected-warning {{use of 'self' after it has been deallocated}}
      // expected-note@-1 {{use of 'self' after it has been deallocated}}
}
@end

@interface SuperDeallocThenCallInstanceMethodClass : NSObject { }
- (void)_invalidate;
@end

@implementation SuperDeallocThenCallInstanceMethodClass
- (void)_invalidate {
}
- (void)dealloc {
  [super dealloc]; // expected-note {{[super dealloc] called here}}
  [self _invalidate]; // expected-warning {{use of 'self' after it has been deallocated}}
      // expected-note@-1 {{use of 'self' after it has been deallocated}}
}
@end

@interface SuperDeallocThenCallNonObjectiveCMethodClass : NSObject { }
@end

static void _invalidate(NSObject *object) {
  (void)object;
}

@implementation SuperDeallocThenCallNonObjectiveCMethodClass
- (void)dealloc {
  [super dealloc]; // expected-note {{[super dealloc] called here}}
  _invalidate(self); // expected-warning {{use of 'self' after it has been deallocated}}
      // expected-note@-1 {{use of 'self' after it has been deallocated}}
}
@end

@interface SuperDeallocThenCallObjectiveClassMethodClass : NSObject { }
@end

@implementation SuperDeallocThenCallObjectiveClassMethodClass
+ (void) invalidate:(id)arg; {
}

- (void)dealloc {
  [super dealloc]; // expected-note {{[super dealloc] called here}}
  [SuperDeallocThenCallObjectiveClassMethodClass invalidate:self]; // expected-warning {{use of 'self' after it has been deallocated}}
      // expected-note@-1 {{use of 'self' after it has been deallocated}}
}
@end

@interface TwoSuperDeallocCallsClass : NSObject {
  NSObject *_ivar;
}
- (void)_invalidate;
@end

@implementation TwoSuperDeallocCallsClass
- (void)_invalidate {
}
- (void)dealloc {
  if (_ivar) { // expected-note {{Taking false branch}}
    [_ivar release];
    [super dealloc];
    return;
  }
  [super dealloc];    // expected-note {{[super dealloc] called here}}
  [self _invalidate]; // expected-warning {{use of 'self' after it has been deallocated}}
      // expected-note@-1 {{use of 'self' after it has been deallocated}}
}
@end

//===------------------------------------------------------------------------===
// Warn about calling [super dealloc] twice due to missing return statement.

@interface MissingReturnCausesDoubleSuperDeallocClass : NSObject {
  NSObject *_ivar;
}
@end

@implementation MissingReturnCausesDoubleSuperDeallocClass
- (void)dealloc {
  if (_ivar) { // expected-note {{Taking true branch}}
    [_ivar release];
    [super dealloc]; // expected-note {{[super dealloc] called here}}
    // return;
  }
  [super dealloc]; // expected-warning{{[super dealloc] should not be called multiple times}}
  // expected-note@-1{{[super dealloc] should not be called multiple times}}
}
@end

//===------------------------------------------------------------------------===
// Warn about calling [super dealloc] twice in two different methods.

@interface SuperDeallocInOtherMethodClass : NSObject {
  NSObject *_ivar;
}
- (void)_cleanup;
@end

@implementation SuperDeallocInOtherMethodClass
- (void)_cleanup {
  [_ivar release];
  [super dealloc]; // expected-note {{[super dealloc] called here}}
}
- (void)dealloc {
  [self _cleanup]; // expected-note {{Calling '_cleanup'}}
  //expected-note@-1 {{Returning from '_cleanup'}}
  [super dealloc]; // expected-warning {{[super dealloc] should not be called multiple times}}
  // expected-note@-1 {{[super dealloc] should not be called multiple times}}
}
@end

//===------------------------------------------------------------------------===
// Do not warn about calling [super dealloc] recursively for different objects
// of the same type with custom retain counting.
//
// A class that contains an ivar of itself with custom retain counting (such
// as provided by _OBJC_SUPPORTED_INLINE_REFCNT_WITH_DEALLOC2MAIN) can generate
// a false positive that [super dealloc] is called twice if each object instance
// is not tracked separately by the checker. This test case is just a simple
// approximation to trigger the false positive.

@class ClassWithOwnIvarInstanceClass;
@interface ClassWithOwnIvarInstanceClass : NSObject {
  ClassWithOwnIvarInstanceClass *_ivar;
  NSUInteger _retainCount;
}
@end

@implementation ClassWithOwnIvarInstanceClass
- (instancetype)retain {
  ++_retainCount;
  return self;
}
- (oneway void)release {
  --_retainCount;
  if (!_retainCount)
    [self dealloc];
}
- (void)dealloc {
  [_ivar release];
  [super dealloc]; // no warning: different instances of same class
}
@end

//===------------------------------------------------------------------------===
// Do not warn about calling [super dealloc] twice if +dealloc is a class
// method.

@interface SuperDeallocClassMethodIgnoredClass : NSObject { }
+ (void)dealloc;
@end

@implementation SuperDeallocClassMethodIgnoredClass
+ (void)dealloc { }
@end

@interface SuperDeallocClassMethodIgnoredSubClass : NSObject { }
+ (void)dealloc;
@end

@implementation SuperDeallocClassMethodIgnoredSubClass
+ (void)dealloc {
  [super dealloc];
  [super dealloc]; // no warning: class method
}
@end

//===------------------------------------------------------------------------===
// Do not warn about calling [super dealloc] twice if when the analyzer has
// inlined the call to its super deallocator.

@interface SuperClassCallingSuperDealloc : NSObject {
  NSObject *_ivar;
}
@end

@implementation SuperClassCallingSuperDealloc
- (void)dealloc; {
  [_ivar release]; // no-warning

  [super dealloc];
}
@end

@interface SubclassCallingSuperDealloc : SuperClassCallingSuperDealloc
@end

@implementation SubclassCallingSuperDealloc
- (void)dealloc; {
  [super dealloc];
}
@end

//===------------------------------------------------------------------------===
// Treat calling [super dealloc] twice as as a sink.

@interface CallingSuperDeallocTwiceIsSink : NSObject
@end

@implementation CallingSuperDeallocTwiceIsSink
- (void)dealloc; {
  [super dealloc]; // expected-note {{[super dealloc] called here}}
  [super dealloc]; // expected-warning {{[super dealloc] should not be called multiple times}}
  // expected-note@-1 {{[super dealloc] should not be called multiple times}}

  clang_analyzer_warnIfReached(); // no-warning
}
@end


//===------------------------------------------------------------------------===
// Test path notes with intervening method call on self.

@interface InterveningMethodCallOnSelf : NSObject
@end

@implementation InterveningMethodCallOnSelf
- (void)anotherMethod {
}

- (void)dealloc; {
  [super dealloc]; // expected-note {{[super dealloc] called here}}
  [self anotherMethod]; // expected-warning {{use of 'self' after it has been deallocated}}
      // expected-note@-1 {{use of 'self' after it has been deallocated}}
  [super dealloc];
}
@end
