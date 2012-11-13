// RUN: %clang --analyze -Xanalyzer -analyzer-checker=osx.cocoa.IncompatibleMethodTypes,osx.coreFoundation.CFRetainRelease -Xclang -verify %s

#include "InlineObjCInstanceMethod.h"

typedef const struct __CFString * CFStringRef;
typedef const void * CFTypeRef;
extern CFTypeRef CFRetain(CFTypeRef cf);
extern void CFRelease(CFTypeRef cf);
extern CFStringRef getString(void);

// Method is defined in the parent; called through self.
@interface MyParent : NSObject
- (int)getInt;
- (const struct __CFString *) testCovariantReturnType __attribute__((cf_returns_retained));
@end
@implementation MyParent
- (int)getInt {
    return 0;
}

- (CFStringRef) testCovariantReturnType __attribute__((cf_returns_retained)) {
  CFStringRef Str = ((void*)0);
  Str = getString();
  if (Str) {
    CFRetain(Str);
  }
  return Str;
}

@end

@interface MyClass : MyParent
@end
@implementation MyClass
- (int)testDynDispatchSelf {
  int y = [self getInt];
  return 5/y; // expected-warning {{Division by zero}}
}

// Get the dynamic type info from a cast (from id to MyClass*).
+ (int)testAllocInit {
  MyClass *a = [[self alloc] init];
  return 5/[a getInt]; // expected-warning {{Division by zero}}
}

// Method is called on inited object.
+ (int)testAllocInit2 {
  MyClass *a = [[MyClass alloc] init];
  return 5/[a getInt]; // expected-warning {{Division by zero}}
}

// Method is called on a parameter.
+ (int)testParam: (MyClass*) a {
  return 5/[a getInt]; // expected-warning {{Division by zero}}
}

// Method is called on a parameter of unnown type.
+ (int)testParamUnknownType: (id) a {
  return 5/[a getInt]; // no warning
}

@end

// TODO: When method is inlined, the attribute reset should be visible.
@interface TestSettingAnAttributeInCallee : NSObject {
  int _attribute;
}
  - (void) method2;
@end

@implementation TestSettingAnAttributeInCallee
- (int) method1 {
  [self method2];
  return 5/_attribute; // expected-warning {{Division by zero}}
}

- (void) method2 {
  _attribute = 0;
}
@end

@interface TestSettingAnAttributeInCaller : NSObject {
  int _attribute;
}
  - (int) method2;
@end

@implementation TestSettingAnAttributeInCaller
- (void) method1 {
  _attribute = 0;
  [self method2];
}

- (int) method2 {
  return 5/_attribute; // expected-warning {{Division by zero}}
}
@end


// Don't crash if we don't know the receiver's region.
void randomlyMessageAnObject(MyClass *arr[], int i) {
  (void)[arr[i] getInt];
}


@interface EvilChild : MyParent
- (id)getInt;
- (const struct __CFString *) testCovariantReturnType __attribute__((cf_returns_retained));
@end

@implementation EvilChild
- (id)getInt { // expected-warning {{types are incompatible}}
  return self;
}
- (CFStringRef) testCovariantReturnType __attribute__((cf_returns_retained)) {
  CFStringRef Str = ((void*)0);
  Str = getString();
  if (Str) {
    CFRetain(Str);
  }
  return Str;
}

@end

int testNonCovariantReturnType() {
  MyParent *obj = [[EvilChild alloc] init];

  // Devirtualization allows us to directly call -[EvilChild getInt], but
  // that returns an id, not an int. There is an off-by-default warning for
  // this, -Woverriding-method-mismatch, and an on-by-default analyzer warning,
  // osx.cocoa.IncompatibleMethodTypes. This code would probably crash at
  // runtime, but at least the analyzer shouldn't crash.
  int x = 1 + [obj getInt];

  [obj release];
  return 5/(x-1); // no-warning
}

int testCovariantReturnTypeNoErrorSinceTypesMatch() {
  MyParent *obj = [[EvilChild alloc] init];

  CFStringRef S = ((void*)0);
  S = [obj testCovariantReturnType];
  if (S)
    CFRelease(S);
  CFRelease(obj);
}
