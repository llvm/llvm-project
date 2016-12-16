// RUN: %clang_cc1 -fblocks -analyze -analyzer-checker=core,nullability.NullPassedToNonnull,nullability.NullReturnedFromNonnull,nullability.NullablePassedToNonnull,nullability.NullableReturnedFromNonnull,nullability.NullableDereferenced -DNOSYSTEMHEADERS=0 -verify %s
// RUN: %clang_cc1 -fblocks -analyze -analyzer-checker=core,nullability.NullPassedToNonnull,nullability.NullReturnedFromNonnull,nullability.NullablePassedToNonnull,nullability.NullableReturnedFromNonnull,nullability.NullableDereferenced -analyzer-config nullability:NoDiagnoseCallsToSystemHeaders=true -DNOSYSTEMHEADERS=1 -verify %s

#include "Inputs/system-header-simulator-for-nullability.h"

@interface TestObject : NSObject
- (int *_Nonnull)returnsNonnull;
- (int *_Nullable)returnsNullable;
- (int *)returnsUnspecified;
- (void)takesNonnull:(int *_Nonnull)p;
- (void)takesNullable:(int *_Nullable)p;
- (void)takesUnspecified:(int *)p;
@property(readonly, strong) NSString *stuff;
@end

TestObject * getUnspecifiedTestObject();
TestObject *_Nonnull getNonnullTestObject();
TestObject *_Nullable getNullableTestObject();

int getRandom();

typedef struct Dummy { int val; } Dummy;

void takesNullable(Dummy *_Nullable);
void takesNonnull(Dummy *_Nonnull);
void takesUnspecified(Dummy *);

Dummy *_Nullable returnsNullable();
Dummy *_Nonnull returnsNonnull();
Dummy *returnsUnspecified();
int *_Nullable returnsNullableInt();

template <typename T> T *eraseNullab(T *p) { return p; }

void takesAttrNonnull(Dummy *p) __attribute((nonnull(1)));

void testBasicRules() {
  Dummy *p = returnsNullable();
  int *ptr = returnsNullableInt();
  // Make every dereference a different path to avoid sinks after errors.
  switch (getRandom()) {
  case 0: {
    Dummy &r = *p; // expected-warning {{Nullable pointer is dereferenced}}
  } break;
  case 1: {
    int b = p->val; // expected-warning {{Nullable pointer is dereferenced}}
  } break;
  case 2: {
    int stuff = *ptr; // expected-warning {{Nullable pointer is dereferenced}}
  } break;
  case 3:
    takesNonnull(p); // expected-warning {{Nullable pointer is passed to a callee that requires a non-null 1st parameter}}
    break;
  case 4: {
    Dummy d;
    takesNullable(&d);
    Dummy dd(d);
    break;
  }
  case 5: takesAttrNonnull(p); break; // expected-warning {{Nullable pointer is passed to}}
  default: { Dummy d = *p; } break; // expected-warning {{Nullable pointer is dereferenced}}
  }
  if (p) {
    takesNonnull(p);
    if (getRandom()) {
      Dummy &r = *p;
    } else {
      int b = p->val;
    }
  }
  Dummy *q = 0;
  if (getRandom()) {
    takesNullable(q);
    takesNonnull(q); // expected-warning {{Null passed to a callee that requires a non-null 1st parameter}}
  }
  Dummy a;
  Dummy *_Nonnull nonnull = &a;
  nonnull = q; // expected-warning {{Null assigned to a pointer which is expected to have non-null value}}
  q = &a;
  takesNullable(q);
  takesNonnull(q);
}

void testMultiParamChecking(Dummy *_Nonnull a, Dummy *_Nullable b,
                            Dummy *_Nonnull c);

void testArgumentTracking(Dummy *_Nonnull nonnull, Dummy *_Nullable nullable) {
  Dummy *p = nullable;
  Dummy *q = nonnull;
  switch(getRandom()) {
  case 1: nonnull = p; break; // expected-warning {{Nullable pointer is assigned to a pointer which is expected to have non-null value}}
  case 2: p = 0; break;
  case 3: q = p; break;
  case 4: testMultiParamChecking(nonnull, nullable, nonnull); break;
  case 5: testMultiParamChecking(nonnull, nonnull, nonnull); break;
  case 6: testMultiParamChecking(nonnull, nullable, nullable); break; // expected-warning {{Nullable pointer is passed to a callee that requires a non-null 3rd parameter}}
  case 7: testMultiParamChecking(nullable, nullable, nonnull); // expected-warning {{Nullable pointer is passed to a callee that requires a non-null 1st parameter}}
  case 8: testMultiParamChecking(nullable, nullable, nullable); // expected-warning {{Nullable pointer is passed to a callee that requires a non-null 1st parameter}}
  case 9: testMultiParamChecking((Dummy *_Nonnull)0, nullable, nonnull); break;
  }
}

Dummy *_Nonnull testNullableReturn(Dummy *_Nullable a) {
  Dummy *p = a;
  return p; // expected-warning {{Nullable pointer is returned from a function that is expected to return a non-null value}}
}

Dummy *_Nonnull testNullReturn() {
  Dummy *p = 0;
  return p; // expected-warning {{Null returned from a function that is expected to return a non-null value}}
}

void testObjCMessageResultNullability() {
  // The expected result: the most nullable of self and method return type.
  TestObject *o = getUnspecifiedTestObject();
  int *shouldBeNullable = [eraseNullab(getNullableTestObject()) returnsNonnull];
  switch (getRandom()) {
  case 0:
    // The core analyzer assumes that the receiver is non-null after a message
    // send. This is to avoid some false positives, and increase performance
    // but it also reduces the coverage and makes this checker unable to reason
    // about the nullness of the receiver. 
    [o takesNonnull:shouldBeNullable]; // No warning expected.
    break;
  case 1:
    shouldBeNullable =
        [eraseNullab(getNullableTestObject()) returnsUnspecified];
    [o takesNonnull:shouldBeNullable]; // No warning expected.
    break;
  case 3:
    shouldBeNullable = [eraseNullab(getNullableTestObject()) returnsNullable];
    [o takesNonnull:shouldBeNullable]; // expected-warning {{Nullable pointer is passed to a callee that requires a non-null 1st parameter}}
    break;
  case 4:
    shouldBeNullable = [eraseNullab(getNonnullTestObject()) returnsNullable];
    [o takesNonnull:shouldBeNullable]; // expected-warning {{Nullable pointer is passed to a callee that requires a non-null 1st parameter}}
    break;
  case 5:
    shouldBeNullable =
        [eraseNullab(getUnspecifiedTestObject()) returnsNullable];
    [o takesNonnull:shouldBeNullable]; // expected-warning {{Nullable pointer is passed to a callee that requires a non-null 1st parameter}}
    break;
  case 6:
    shouldBeNullable = [eraseNullab(getNullableTestObject()) returnsNullable];
    [o takesNonnull:shouldBeNullable]; // expected-warning {{Nullable pointer is passed to a callee that requires a non-null 1st parameter}}
    break;
  case 7: {
    int *shouldBeNonnull = [eraseNullab(getNonnullTestObject()) returnsNonnull];
    [o takesNonnull:shouldBeNonnull];
  } break;
  }
}

Dummy * _Nonnull testDirectCastNullableToNonnull() {
  Dummy *p = returnsNullable();
  takesNonnull((Dummy * _Nonnull)p);  // no-warning
  return (Dummy * _Nonnull)p;         // no-warning
}

Dummy * _Nonnull testIndirectCastNullableToNonnull() {
  Dummy *p = (Dummy * _Nonnull)returnsNullable();
  takesNonnull(p);  // no-warning
  return p;         // no-warning
}

Dummy * _Nonnull testDirectCastNilToNonnull() {
  takesNonnull((Dummy * _Nonnull)0);  // no-warning
  return (Dummy * _Nonnull)0;         // no-warning
}

void testIndirectCastNilToNonnullAndPass() {
  Dummy *p = (Dummy * _Nonnull)0;
  // FIXME: Ideally the cast above would suppress this warning.
  takesNonnull(p);  // expected-warning {{Null passed to a callee that requires a non-null 1st parameter}}
}

void testDirectCastNilToNonnullAndAssignToLocalInInitializer() {
  Dummy * _Nonnull nonnullLocalWithAssignmentInInitializer = (Dummy * _Nonnull)0; // no-warning
  (void)nonnullLocalWithAssignmentInInitializer;

  // Since we've already had an invariant violation along this path,
  // we shouldn't warn here.
  nonnullLocalWithAssignmentInInitializer = 0;
  (void)nonnullLocalWithAssignmentInInitializer;

}

void testDirectCastNilToNonnullAndAssignToLocal(Dummy * _Nonnull p) {
  Dummy * _Nonnull nonnullLocalWithAssignment = p;
  nonnullLocalWithAssignment = (Dummy * _Nonnull)0; // no-warning
  (void)nonnullLocalWithAssignment;

  // Since we've already had an invariant violation along this path,
  // we shouldn't warn here.
  nonnullLocalWithAssignment = 0;
  (void)nonnullLocalWithAssignment;
}

void testDirectCastNilToNonnullAndAssignToParam(Dummy * _Nonnull p) {
  p = (Dummy * _Nonnull)0; // no-warning
}

@interface ClassWithNonnullIvar : NSObject {
  Dummy *_nonnullIvar;
}
@end

@implementation ClassWithNonnullIvar
-(void)testDirectCastNilToNonnullAndAssignToIvar {
  _nonnullIvar = (Dummy * _Nonnull)0; // no-warning;

  // Since we've already had an invariant violation along this path,
  // we shouldn't warn here.
  _nonnullIvar = 0;
}
@end

void testIndirectNilPassToNonnull() {
  Dummy *p = 0;
  takesNonnull(p);  // expected-warning {{Null passed to a callee that requires a non-null 1st parameter}}
}

void testConditionalNilPassToNonnull(Dummy *p) {
  if (!p) {
    takesNonnull(p);  // expected-warning {{Null passed to a callee that requires a non-null 1st parameter}}
  }
}

Dummy * _Nonnull testIndirectCastNilToNonnullAndReturn() {
  Dummy *p = (Dummy * _Nonnull)0;
  // FIXME: Ideally the cast above would suppress this warning.
  return p; // expected-warning {{Null returned from a function that is expected to return a non-null value}}
}

void testInvalidPropagation() {
  Dummy *p = returnsUnspecified();
  takesNullable(p);
  takesNonnull(p);
}

void onlyReportFirstPreconditionViolationOnPath() {
  Dummy *p = returnsNullable();
  takesNonnull(p); // expected-warning {{Nullable pointer is passed to a callee that requires a non-null 1st parameter}}
  takesNonnull(p); // No warning.
  // The first warning was not a sink. The analysis expected to continue.
  int i = 0;
  i = 5 / i; // expected-warning {{Division by zero}}
  (void)i;
}

Dummy *_Nonnull doNotWarnWhenPreconditionIsViolatedInTopFunc(
    Dummy *_Nonnull p) {
  if (!p) {
    Dummy *ret =
        0; // avoid compiler warning (which is not generated by the analyzer)
    if (getRandom())
      return ret; // no warning
    else
      return p; // no warning
  } else {
    return p;
  }
}

Dummy *_Nonnull doNotWarnWhenPreconditionIsViolated(Dummy *_Nonnull p) {
  if (!p) {
    Dummy *ret =
        0; // avoid compiler warning (which is not generated by the analyzer)
    if (getRandom())
      return ret; // no warning
    else
      return p; // no warning
  } else {
    return p;
  }
}

void testPreconditionViolationInInlinedFunction(Dummy *p) {
  doNotWarnWhenPreconditionIsViolated(p);
}

@interface TestInlinedPreconditionViolationClass : NSObject
@end

@implementation TestInlinedPreconditionViolationClass
-(Dummy * _Nonnull) calleeWithParam:(Dummy * _Nonnull) p2 {
  Dummy *x = 0;
  if (!p2) // p2 binding becomes dead at this point.
    return x; // no-warning
  else
   return p2;
}

-(Dummy *)callerWithParam:(Dummy * _Nonnull) p1 {
  return [self calleeWithParam:p1];
}

@end

int * _Nonnull InlinedPreconditionViolationInFunctionCallee(int * _Nonnull p2) {
  int *x = 0;
  if (!p2) // p2 binding becomes dead at this point.
    return x; // no-warning
  else
   return p2;
}

int * _Nonnull InlinedReturnNullOverSuppressionCallee(int * _Nonnull p2) {
  int *result = 0;
  return result; // no-warning; but this is an over suppression
}

int *InlinedReturnNullOverSuppressionCaller(int * _Nonnull p1) {
  return InlinedReturnNullOverSuppressionCallee(p1);
}

void inlinedNullable(Dummy *_Nullable p) {
  if (p) return;
}
void inlinedNonnull(Dummy *_Nonnull p) {
  if (p) return;
}
void inlinedUnspecified(Dummy *p) {
  if (p) return;
}

void testNilReturnWithBlock(Dummy *p) {
  p = 0;
  Dummy *_Nonnull (^myblock)(void) = ^Dummy *_Nonnull(void) {
    return p; // TODO: We should warn in blocks.
  };
  myblock();
}

Dummy *_Nonnull testDefensiveInlineChecks(Dummy * p) {
  switch (getRandom()) {
  case 1: inlinedNullable(p); break;
  case 2: inlinedNonnull(p); break;
  case 3: inlinedUnspecified(p); break;
  }
  if (getRandom())
    takesNonnull(p);  // no-warning

  if (getRandom()) {
    Dummy *_Nonnull varWithInitializer = p; // no-warning

     Dummy *_Nonnull var1WithInitializer = p,  // no-warning
           *_Nonnull var2WithInitializer = p;  // no-warning
  }

  if (getRandom()) {
    Dummy *_Nonnull varWithoutInitializer;
    varWithoutInitializer = p; // no-warning
  }

  return p;
}


@interface SomeClass : NSObject {
  int instanceVar;
}
@end

@implementation SomeClass (MethodReturn)
- (id)initWithSomething:(int)i {
  if (self = [super init]) {
    instanceVar = i;
  }

  return self;
}

- (TestObject * _Nonnull)testReturnsNullableInNonnullIndirectly {
  TestObject *local = getNullableTestObject();
  return local; // expected-warning {{Nullable pointer is returned from a method that is expected to return a non-null value}}
}

- (TestObject * _Nonnull)testReturnsCastSuppressedNullableInNonnullIndirectly {
  TestObject *local = getNullableTestObject();
  return (TestObject * _Nonnull)local; // no-warning
}

- (TestObject * _Nonnull)testReturnsNullableInNonnullWhenPreconditionViolated:(TestObject * _Nonnull) p {
  TestObject *local = getNullableTestObject();
  if (!p) // Pre-condition violated here.
    return local; // no-warning
  else
    return p; // no-warning
}
@end

@interface ClassWithInitializers : NSObject
@end

@implementation ClassWithInitializers
- (instancetype _Nonnull)initWithNonnullReturnAndSelfCheckingIdiom {
  // This defensive check is a common-enough idiom that we filter don't want
  // to issue a diagnostic for it,
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
@end

@interface SubClassWithInitializers : ClassWithInitializers
@end

@implementation SubClassWithInitializers
// Note: Because this is overridding
// -[ClassWithInitializers initWithNonnullReturnAndSelfCheckingIdiom],
// the return type of this method becomes implicitly id _Nonnull.
- (id)initWithNonnullReturnAndSelfCheckingIdiom {
  if (self = [super initWithNonnullReturnAndSelfCheckingIdiom]) {
  }

  return self; // no-warning
}

- (id _Nonnull)initWithNonnullReturnAndSelfCheckingIdiomV2; {
  // Another common return-checking idiom
  self = [super initWithNonnullReturnAndSelfCheckingIdiom];
  if (!self) {
    return nil; // no-warning
  }

  return self;
}
@end

@interface ClassWithCopyWithZone : NSObject<NSCopying,NSMutableCopying> {
  id i;
}

@end

@implementation ClassWithCopyWithZone
-(id)copyWithZone:(NSZone *)zone {
  ClassWithCopyWithZone *newInstance = [[ClassWithCopyWithZone alloc] init];
  if (!newInstance)
    return nil;

  newInstance->i = i;
  return newInstance;
}

-(id)mutableCopyWithZone:(NSZone *)zone {
  ClassWithCopyWithZone *newInstance = [[ClassWithCopyWithZone alloc] init];
  if (newInstance) {
    newInstance->i = i;
  }

  return newInstance;
}
@end

NSString * _Nullable returnsNullableString();

void callFunctionInSystemHeader() {
  NSString *s = returnsNullableString();

  NSSystemFunctionTakingNonnull(s);
  #if !NOSYSTEMHEADERS
  // expected-warning@-2{{Nullable pointer is passed to a callee that requires a non-null 1st parameter}}
  #endif
}

void callMethodInSystemHeader() {
  NSString *s = returnsNullableString();

  NSSystemClass *sc = [[NSSystemClass alloc] init];
  [sc takesNonnull:s];
  #if !NOSYSTEMHEADERS
  // expected-warning@-2{{Nullable pointer is passed to a callee that requires a non-null 1st parameter}}
  #endif
}

// Test to make sure the analyzer doesn't warn when an a nullability invariant
// has already been found to be violated on an instance variable.

@class MyInternalClass;
@interface MyClass : NSObject {
  MyInternalClass * _Nonnull _internal;
}
@end

@interface MyInternalClass : NSObject {
  @public
  id _someIvar;
}
-(id _Nonnull)methodWithInternalImplementation;
@end

@interface MyClass () {
  MyInternalClass * _Nonnull _nilledOutInternal;
}
@end

@implementation MyClass
-(id _Nonnull)methodWithInternalImplementation {
  if (!_internal)
    return nil; // no-warning

  return [_internal methodWithInternalImplementation];
}

- (id _Nonnull)methodReturningIvarInImplementation; {
  return _internal == 0 ? nil : _internal->_someIvar; // no-warning
}

-(id _Nonnull)methodWithNilledOutInternal {
  _nilledOutInternal = (id _Nonnull)nil;

  return nil; // no-warning
}
@end
