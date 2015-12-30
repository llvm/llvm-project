// RUN: %clang_cc1 -fobjc-arc -analyze -analyzer-checker=core,nullability -verify %s

#define nil 0
#define BOOL int

@protocol NSObject
+ (id)alloc;
- (id)init;
@end

@protocol NSCopying
@end

__attribute__((objc_root_class))
@interface
NSObject<NSObject>
@end

@interface NSString : NSObject<NSCopying>
- (BOOL)isEqualToString : (NSString *_Nonnull)aString;
- (NSString *)stringByAppendingString:(NSString *_Nonnull)aString;
@end

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
    Dummy &r = *p; // expected-warning {{}}
  } break;
  case 1: {
    int b = p->val; // expected-warning {{}}
  } break;
  case 2: {
    int stuff = *ptr; // expected-warning {{}}
  } break;
  case 3:
    takesNonnull(p); // expected-warning {{}}
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
    takesNonnull(q); // expected-warning {{}}
  }
  Dummy a;
  Dummy *_Nonnull nonnull = &a;
  nonnull = q; // expected-warning {{}}
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
  case 1: nonnull = p; break; // expected-warning {{}}
  case 2: p = 0; break;
  case 3: q = p; break;
  case 4: testMultiParamChecking(nonnull, nullable, nonnull); break;
  case 5: testMultiParamChecking(nonnull, nonnull, nonnull); break;
  case 6: testMultiParamChecking(nonnull, nullable, nullable); break; // expected-warning {{}}
  case 7: testMultiParamChecking(nullable, nullable, nonnull); // expected-warning {{}}
  case 8: testMultiParamChecking(nullable, nullable, nullable); // expected-warning {{}}
  case 9: testMultiParamChecking((Dummy *_Nonnull)0, nullable, nonnull); break;
  }
}

Dummy *_Nonnull testNullableReturn(Dummy *_Nullable a) {
  Dummy *p = a;
  return p; // expected-warning {{}}
}

Dummy *_Nonnull testNullReturn() {
  Dummy *p = 0;
  return p; // expected-warning {{}}
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
    [o takesNonnull:shouldBeNullable]; // expected-warning {{}}
    break;
  case 4:
    shouldBeNullable = [eraseNullab(getNonnullTestObject()) returnsNullable];
    [o takesNonnull:shouldBeNullable]; // expected-warning {{}}
    break;
  case 5:
    shouldBeNullable =
        [eraseNullab(getUnspecifiedTestObject()) returnsNullable];
    [o takesNonnull:shouldBeNullable]; // expected-warning {{}}
    break;
  case 6:
    shouldBeNullable = [eraseNullab(getNullableTestObject()) returnsNullable];
    [o takesNonnull:shouldBeNullable]; // expected-warning {{}}
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
  takesNonnull(p);  // expected-warning {{Null passed to a callee that requires a non-null argument}}
}

Dummy * _Nonnull testIndirectCastNilToNonnullAndReturn() {
  Dummy *p = (Dummy * _Nonnull)0;
  // FIXME: Ideally the cast above would suppress this warning.
  return p; // expected-warning {{Null is returned from a function that is expected to return a non-null value}}
}

void testInvalidPropagation() {
  Dummy *p = returnsUnspecified();
  takesNullable(p);
  takesNonnull(p);
}

void onlyReportFirstPreconditionViolationOnPath() {
  Dummy *p = returnsNullable();
  takesNonnull(p); // expected-warning {{}}
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

void inlinedNullable(Dummy *_Nullable p) {
  if (p) return;
}
void inlinedNonnull(Dummy *_Nonnull p) {
  if (p) return;
}
void inlinedUnspecified(Dummy *p) {
  if (p) return;
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

void testObjCARCImplicitZeroInitialization() {
  TestObject * _Nonnull implicitlyZeroInitialized; // no-warning
  implicitlyZeroInitialized = getNonnullTestObject();
}

void testObjCARCExplicitZeroInitialization() {
  TestObject * _Nonnull explicitlyZeroInitialized = nil; // expected-warning {{Null is assigned to a pointer which is expected to have non-null value}}
}
