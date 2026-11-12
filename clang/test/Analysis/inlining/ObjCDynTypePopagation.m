// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config ipa=dynamic-bifurcate -verify %s

#include "InlineObjCInstanceMethod.h"

void clang_analyzer_eval(int);

PublicSubClass2 *getObj(void);

@implementation PublicParent
- (int)getZeroOverridden {
  return 1;
}
- (int)getZero {
  return 0;
}
@end

@implementation PublicSubClass2
- (int)getZeroOverridden {
  return 0;
}

/* Test that we get the right type from call to alloc. */
+ (void)testAllocSelf {
  id a = [self alloc];
  clang_analyzer_eval([a getZeroOverridden] == 0); // expected-warning{{TRUE}}
}

+ (void)testAllocClass {
  id a = [PublicSubClass2 alloc];
  clang_analyzer_eval([a getZeroOverridden] == 0); // expected-warning{{TRUE}}
}

+ (void)testAllocSuperOverriden {
  id a = [super alloc];
  // Evaluates to 1 in the parent.
  clang_analyzer_eval([a getZeroOverridden] == 0); // expected-warning{{FALSE}}
}

+ (void)testAllocSuper {
  id a = [super alloc];
  clang_analyzer_eval([a getZero] == 0); // expected-warning{{TRUE}}
}

+ (void)testAllocInit {
  id a = [[self alloc] init];
  clang_analyzer_eval([a getZeroOverridden] == 0); // expected-warning{{TRUE}}
}

+ (void)testNewSelf {
  id a = [self new];
  clang_analyzer_eval([a getZeroOverridden] == 0); // expected-warning{{TRUE}}
}

// Casting to parent should not pessimize the dynamic type.
+ (void)testCastToParent {
  id a = [[self alloc] init];
  PublicParent *p = a;
  clang_analyzer_eval([p getZeroOverridden] == 0); // expected-warning{{TRUE}}
}

// The type of parameter gets used.
+ (void)testTypeFromParam:(PublicParent *)p {
  clang_analyzer_eval([p getZero] == 0); // expected-warning{{TRUE}}
}

// Test implicit cast.
// Note, in this case, p could also be a subclass of MyParent.
+ (void)testCastFromId:(id)a {
  PublicParent *p = a;
  clang_analyzer_eval([p getZero] == 0); // expected-warning{{TRUE}}
}
@end

// TODO: Would be nice to handle the case of dynamically obtained class info
// as well. We need a MemRegion for class types for this.
int testDynamicClass(BOOL coin) {
  Class AllocClass = (coin ? [NSObject class] : [PublicSubClass2 class]);
  id x = [[AllocClass alloc] init];
  if (coin)
    return [x getZero];
  return 1;
}

@interface UserClass : NSObject
- (PublicSubClass2 *)_newPublicSubClass2;
- (int)getZero;
- (void)callNew;
@end

@implementation UserClass
- (PublicSubClass2 *)_newPublicSubClass2 {
  return [[PublicSubClass2 alloc] init];
}
- (int)getZero {
  return 5;
}
- (void)callNew {
  PublicSubClass2 *x = [self _newPublicSubClass2];
  clang_analyzer_eval([x getZero] == 0); //expected-warning{{TRUE}}
}
@end