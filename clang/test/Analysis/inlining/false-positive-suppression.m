// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-config suppress-null-return-paths=false -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify -DSUPPRESSED=1 %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -fobjc-arc -verify -DSUPPRESSED=1 %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-config avoid-suppressing-null-argument-paths=true -DSUPPRESSED=1 -DNULL_ARGS=1 -verify %s

#define ARC __has_feature(objc_arc)

#ifdef SUPPRESSED
// expected-no-diagnostics
#endif

@interface PointerWrapper
- (int *)getPtr;
- (id)getObject;
@end

id getNil() {
  return 0;
}

void testNilReceiverHelperA(int *x) {
  *x = 1;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}

void testNilReceiverHelperB(int *x) {
  *x = 1;
#if !defined(SUPPRESSED)
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}

void testNilReceiver(int coin) {
  id x = getNil();
  if (coin)
    testNilReceiverHelperA([x getPtr]);
  else
    testNilReceiverHelperB([[x getObject] getPtr]);
}

// FALSE NEGATIVES (over-suppression)

__attribute__((objc_root_class))
@interface SomeClass {
  int ivar;
}
-(int *)methodReturningNull;

@property(readonly) int *propertyReturningNull;

@property(readonly) int *synthesizedProperty;

@property(readonly) SomeClass *propertyReturningNil;

@end

@interface SubOfSomeClass : SomeClass
@end

@implementation SubOfSomeClass
@end

@implementation SomeClass
-(int *)methodReturningNull {
  return 0;
}

-(int *)propertyReturningNull {
  return 0;
}

-(SomeClass *)propertyReturningNil {
  return 0;
}

+(int *)classPropertyReturningNull {
  return 0;
}
@end

void testMethodReturningNull(SomeClass *sc) {
  int *result = [sc methodReturningNull];
  *result = 1;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}

void testPropertyReturningNull(SomeClass *sc) {
  int *result = sc.propertyReturningNull;
  *result = 1;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}

@implementation SubOfSomeClass (ForTestOfSuperProperty)
-(void)testSuperPropertyReturningNull {
  int *result = super.propertyReturningNull;
  *result = 1;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}
@end

void testClassPropertyReturningNull() {
  int *result = SomeClass.classPropertyReturningNull;
  *result = 1;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}

@implementation SomeClass (ForTestOfPropertyReturningNil)
void testPropertyReturningNil(SomeClass *sc) {
  SomeClass *result = sc.propertyReturningNil;
  result->ivar = 1;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Access to instance variable 'ivar' results in a dereference of a null pointer (loaded from variable 'result')}}
#endif
}
@end

void testSynthesizedPropertyReturningNull(SomeClass *sc) {
  if (sc.synthesizedProperty)
    return;

  int *result = sc.synthesizedProperty;
  *result = 1;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}
