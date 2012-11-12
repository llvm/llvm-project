// RUN: %clang_cc1 -analyze -analyzer-checker=core,osx -analyzer-ipa=dynamic-bifurcate -verify %s

#include "InlineObjCInstanceMethod.h"

@interface MyParent : NSObject
- (int)getZero;
@end
@implementation MyParent
- (int)getZero {
    return 0;
}
@end

@interface PublicClass () {
   int value2;
}
@property (readwrite) int value1;
- (void)setValue2:(int)newValue2;
@end

@implementation PublicClass

- (int)getZeroPublic {
    return 0;
}

@synthesize value1;

- (int)value2 {
    return value2;
} 
- (void)setValue2:(int)newValue {
    value2 = newValue;
}

- (int)value3 {
    return value3;
} 
- (void)setValue3:(int)newValue {
    value3 = newValue;
}

@end

@interface MyClassWithPublicParent : PublicClass
- (int)getZeroPublic;
@end
@implementation MyClassWithPublicParent
- (int)getZeroPublic {
    return 0;
}
@end

// Category overrides a public method.
@interface PublicSubClass (PrvateCat)
  - (int) getZeroPublic;
@end
@implementation PublicSubClass (PrvateCat)
- (int)getZeroPublic {
    return 0;
}
@end


@interface MyClass : MyParent {
  int value;
}
- (int)getZero;
@property int value;
@end

// Since class is private, we assume that it cannot be subclassed.
// False negative: this class is "privately subclassed". this is very rare 
// in practice.
@implementation MyClass
+ (int) testTypeFromParam:(MyParent*) p {
  int m = 0;
  int z = [p getZero];
  if (z)
    return 5/m; // false negative
  return 5/[p getZero];// expected-warning {{Division by zero}}
}

// Here only one definition is possible, since the declaration is not visible 
// from outside. 
+ (int) testTypeFromParamPrivateChild:(MyClass*) c {
  int m = 0;
  int z = [c getZero]; // MyClass overrides getZero to return '1'.
  if (z)
    return 5/m; // expected-warning {{Division by zero}}
  return 5/[c getZero];//no warning
}

- (int)getZero {
    return 1;
}

- (int)value {
  return value;
}
 
- (void)setValue:(int)newValue {
  value = newValue;
}

// Test ivar access.
- (int) testIvarInSelf {
  value = 0;
  return 5/value; // expected-warning {{Division by zero}}
}

+ (int) testIvar: (MyClass*) p {
  p.value = 0;
  return 5/p.value; // expected-warning {{Division by zero}}
}

// Test simple property access.
+ (int) testProperty: (MyClass*) p {
  int x= 0;
  [p setValue:0];
  return 5/[p value]; // expected-warning {{Division by zero}}  
}

@end

// The class is prvate and is not subclassed.
int testCallToPublicAPIInParent(MyClassWithPublicParent *p) {
  int m = 0;
  int z = [p getZeroPublic];
  if (z)
    return 5/m; // no warning
  return 5/[p getZeroPublic];// expected-warning {{Division by zero}}  
}

// When the called method is public (due to it being defined outside of main file),
// split the path and analyze both branches.
// In this case, p can be either the object of type MyParent* or MyClass*:
// - If it's MyParent*, getZero returns 0.
// - If it's MyClass*, getZero returns 1 and 'return 5/m' is reachable.
// Declaration is provate, but p can be a subclass (MyClass*).
int testCallToPublicAPI(PublicClass *p) {
  int m = 0;
  int z = [p getZeroPublic];
  if (z)
    return 5/m; // expected-warning {{Division by zero}}
  return 5/[p getZeroPublic];// expected-warning {{Division by zero}}  
}

// Even though the method is privately declared in the category, the parent 
// declares the method as public. Assume the instance can be subclassed.
int testCallToPublicAPICat(PublicSubClass *p) {
  int m = 0;
  int z = [p getZeroPublic];
  if (z)
    return 5/m; // expected-warning {{Division by zero}}
  return 5/[p getZeroPublic];// expected-warning {{Division by zero}}  
}

// Test public property - properties should always be inlined, regardless 
// weither they are "public" or private. 
int testPublicProperty(PublicClass *p) {
  int x = 0;
  p.value3 = 0;
  if (p.value3 != 0)
    return 5/x; 
  return 5/p.value3;// expected-warning {{Division by zero}}
}

int testExtension(PublicClass *p) {
  int x = 0;
  [p setValue2:0];
  if ([p value2] != 0)
    return 5/x; // expected-warning {{Division by zero}}
  return 5/[p value2]; // expected-warning {{Division by zero}}
}

// TODO: we do not handle synthesized properties yet.
int testPropertySynthesized(PublicClass *p) {
  [p setValue1:0];
  return 5/[p value1];  
}

// Test definition not available edge case.
@interface DefNotAvailClass : NSObject
@end
id testDefNotAvailableInlined(DefNotAvailClass *C) {
  return [C mem]; // expected-warning {{instance method '-mem' not found}}
}
id testDefNotAvailable(DefNotAvailClass *C) {
  return testDefNotAvailableInlined(C);
}