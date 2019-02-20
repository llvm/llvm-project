// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core,osx.coreFoundation.CFRetainRelease,osx.cocoa.ClassRelease,osx.cocoa.RetainCount -analyzer-store=region -analyzer-output=text -verify %s
// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core,osx.coreFoundation.CFRetainRelease,osx.cocoa.ClassRelease,osx.cocoa.RetainCount -analyzer-store=region -analyzer-output=plist-multi-file %s -o %t
// RUN: cat %t | %diff_plist %S/Inputs/expected-plists/retain-release-path-notes.m.plist -

/***
This file is for testing the path-sensitive notes for retain/release errors.
Its goal is to have simple branch coverage of any path-based diagnostics,
not to actually check all possible retain/release errors.

This file includes notes that only appear in a ref-counted analysis. 
GC-specific notes should go in retain-release-path-notes-gc.m.
***/

@interface NSObject
+ (id)alloc;
- (id)init;
- (void)dealloc;

- (Class)class;

- (id)retain;
- (void)release;
- (void)autorelease;
@end

@interface Foo : NSObject
- (id)methodWithValue;
@property(retain) id propertyValue;

- (id)objectAtIndexedSubscript:(unsigned)index;
- (id)objectForKeyedSubscript:(id)key;
@end

typedef struct CFType *CFTypeRef;
CFTypeRef CFRetain(CFTypeRef);
void CFRelease(CFTypeRef);
CFTypeRef CFAutorelease(CFTypeRef __attribute__((cf_consumed)));

id NSMakeCollectable(CFTypeRef);
CFTypeRef CFMakeCollectable(CFTypeRef);

CFTypeRef CFCreateSomething();
CFTypeRef CFGetSomething();


void creationViaAlloc () {
  id leaked = [[NSObject alloc] init]; // expected-note{{Method returns an instance of NSObject with a +1 retain count}}
  return; // expected-warning{{leak}} expected-note{{Object leaked: object allocated and stored into 'leaked' is not referenced later in this execution path and has a retain count of +1}}
}

void creationViaCFCreate () {
  CFTypeRef leaked = CFCreateSomething(); // expected-note{{Call to function 'CFCreateSomething' returns a Core Foundation object of type 'CFTypeRef' with a +1 retain count}}
  return; // expected-warning{{leak}} expected-note{{Object leaked: object allocated and stored into 'leaked' is not referenced later in this execution path and has a retain count of +1}}
}

void acquisitionViaMethod (Foo *foo) {
  id leaked = [foo methodWithValue]; // expected-note{{Method returns an Objective-C object with a +0 retain count}}
  [leaked retain]; // expected-note{{Reference count incremented. The object now has a +1 retain count}}
  [leaked retain]; // expected-note{{Reference count incremented. The object now has a +2 retain count}}
  [leaked release]; // expected-note{{Reference count decremented. The object now has a +1 retain count}}
  return; // expected-warning{{leak}} expected-note{{Object leaked: object allocated and stored into 'leaked' is not referenced later in this execution path and has a retain count of +1}}
}

void acquisitionViaProperty (Foo *foo) {
  id leaked = foo.propertyValue; // expected-note{{Property returns an Objective-C object with a +0 retain count}}
  [leaked retain]; // expected-note{{Reference count incremented. The object now has a +1 retain count}}
  return; // expected-warning{{leak}} expected-note{{Object leaked: object allocated and stored into 'leaked' is not referenced later in this execution path and has a retain count of +1}}
}

void acquisitionViaCFFunction () {
  CFTypeRef leaked = CFGetSomething(); // expected-note{{Call to function 'CFGetSomething' returns a Core Foundation object of type 'CFTypeRef' with a +0 retain count}}
  CFRetain(leaked); // expected-note{{Reference count incremented. The object now has a +1 retain count}}
  return; // expected-warning{{leak}} expected-note{{Object leaked: object allocated and stored into 'leaked' is not referenced later in this execution path and has a retain count of +1}}
}

void explicitDealloc () {
  id object = [[NSObject alloc] init]; // expected-note{{Method returns an instance of NSObject with a +1 retain count}}
  [object dealloc]; // expected-note{{Object released by directly sending the '-dealloc' message}}
  [object class]; // expected-warning{{Reference-counted object is used after it is released}} // expected-note{{Reference-counted object is used after it is released}}
}

void implicitDealloc () {
  id object = [[NSObject alloc] init]; // expected-note{{Method returns an instance of NSObject with a +1 retain count}}
  [object release]; // expected-note{{Object released}}
  [object class]; // expected-warning{{Reference-counted object is used after it is released}} // expected-note{{Reference-counted object is used after it is released}}
}

void overAutorelease () {
  id object = [[NSObject alloc] init]; // expected-note{{Method returns an instance of NSObject with a +1 retain count}}
  [object autorelease]; // expected-note{{Object autoreleased}}
  [object autorelease]; // expected-note{{Object autoreleased}} 
  return; // expected-warning{{Object autoreleased too many times}} expected-note{{Object was autoreleased 2 times but the object has a +1 retain count}} 
}

void autoreleaseUnowned (Foo *foo) {
  id object = foo.propertyValue; // expected-note{{Property returns an Objective-C object with a +0 retain count}}
  [object autorelease]; // expected-note{{Object autoreleased}} 
  return; // expected-warning{{Object autoreleased too many times}} expected-note{{Object was autoreleased but has a +0 retain count}}
}

void makeCollectableIgnored() {
  CFTypeRef leaked = CFCreateSomething(); // expected-note{{Call to function 'CFCreateSomething' returns a Core Foundation object of type 'CFTypeRef' with a +1 retain count}}
  CFMakeCollectable(leaked);
  NSMakeCollectable(leaked);
  return; // expected-warning{{leak}} expected-note{{Object leaked: object allocated and stored into 'leaked' is not referenced later in this execution path and has a retain count of +1}}
}

CFTypeRef CFCopyRuleViolation () {
  CFTypeRef object = CFGetSomething(); // expected-note{{Call to function 'CFGetSomething' returns a Core Foundation object of type 'CFTypeRef' with a +0 retain count}}
  return object; // expected-warning{{Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected}} expected-note{{Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected}}
}

CFTypeRef CFGetRuleViolation () {
  CFTypeRef object = CFCreateSomething(); // expected-note{{Call to function 'CFCreateSomething' returns a Core Foundation object of type 'CFTypeRef' with a +1 retain count}}
  return object; // expected-warning{{leak}} expected-note{{Object leaked: object allocated and stored into 'object' is returned from a function whose name ('CFGetRuleViolation') does not contain 'Copy' or 'Create'.  This violates the naming convention rules given in the Memory Management Guide for Core Foundation}}
}

@implementation Foo (FundamentalMemoryManagementRules)
- (id)copyViolation {
  id result = self.propertyValue; // expected-note{{Property returns an Objective-C object with a +0 retain count}}
  return result; // expected-warning{{Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected}} expected-note{{Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected}}
}

- (id)copyViolationIndexedSubscript {
  id result = self[0]; // expected-note{{Subscript returns an Objective-C object with a +0 retain count}}
  return result; // expected-warning{{Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected}} expected-note{{Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected}}
}

- (id)copyViolationKeyedSubscript {
  id result = self[self]; // expected-note{{Subscript returns an Objective-C object with a +0 retain count}}
  return result; // expected-warning{{Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected}} expected-note{{Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected}}
}

- (id)getViolation {
  id result = [[Foo alloc] init]; // expected-note{{Method returns an instance of Foo with a +1 retain count}}
  return result; // expected-warning{{leak}} expected-note{{Object leaked: object allocated and stored into 'result' is returned from a method whose name ('getViolation') does not start with 'copy', 'mutableCopy', 'alloc' or 'new'.  This violates the naming convention rules given in the Memory Management Guide for Cocoa}}
}

- (id)copyAutorelease {
  id result = [[Foo alloc] init]; // expected-note{{Method returns an instance of Foo with a +1 retain count}}
  [result autorelease]; // expected-note{{Object autoreleased}}
  return result; // expected-warning{{Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected}} expected-note{{Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected}}
}
@end


typedef unsigned long NSUInteger;

@interface NSValue : NSObject
@end

@interface NSNumber : NSValue
+ (NSNumber *)numberWithInt:(int)i;
@end

@interface NSString : NSObject
+ (NSString *)stringWithUTF8String:(const char *)str;
@end

@interface NSArray : NSObject
+ (NSArray *)arrayWithObjects:(const id [])objects count:(NSUInteger)count;
@end

@interface NSDictionary : NSObject
+ (id)dictionaryWithObjects:(const id [])objects forKeys:(const id /* <NSCopying> */ [])keys count:(NSUInteger)count;
@end


void testNumericLiteral() {
  id result = @1; // expected-note{{NSNumber literal is an object with a +0 retain count}}
  [result release]; // expected-warning{{decrement}} expected-note{{Incorrect decrement of the reference count of an object that is not owned at this point by the caller}}
}

void testBoxedInt(int x) {
  id result = @(x); // expected-note{{NSNumber boxed expression produces an object with a +0 retain count}}
  [result release]; // expected-warning{{decrement}} expected-note{{Incorrect decrement of the reference count of an object that is not owned at this point by the caller}}
}

void testBoxedString(const char *str) {
  id result = @(str); // expected-note{{NSString boxed expression produces an object with a +0 retain count}}
  [result release]; // expected-warning{{decrement}} expected-note{{Incorrect decrement of the reference count of an object that is not owned at this point by the caller}}
}

void testArray(id obj) {
  id result = @[obj]; // expected-note{{NSArray literal is an object with a +0 retain count}}
  [result release]; // expected-warning{{decrement}} expected-note{{Incorrect decrement of the reference count of an object that is not owned at this point by the caller}}
}

void testDictionary(id key, id value) {
  id result = @{key: value}; // expected-note{{NSDictionary literal is an object with a +0 retain count}}
  [result release]; // expected-warning{{decrement}} expected-note{{Incorrect decrement of the reference count of an object that is not owned at this point by the caller}}
}

// Test that we step into the init method when the allocated object is leaked due to early escape within init.

static int Cond;
@interface MyObj : NSObject
-(id)initX;
-(id)initY;
-(id)initZ;
+(void)test;
@end

@implementation MyObj

-(id)initX {
  if (Cond)  // expected-note {{Assuming 'Cond' is not equal to 0}}
             // expected-note@-1{{Taking true branch}}
    return 0;
  self = [super init];
  return self;
}

-(id)initY {
  self = [super init]; //expected-note {{Method returns an instance of MyObj with a +1 retain count}}
  return self;
}

-(id)initZ {
  self = [super init];
  return self;
}

+(void)test {
  // initX is inlined since we explicitly mark it as interesting
  id x = [[MyObj alloc] initX]; // expected-warning {{Potential leak of an object}}
                                // expected-note@-1 {{Method returns an instance of MyObj with a +1 retain count}}
                                // expected-note@-2 {{Calling 'initX'}}
                                // expected-note@-3 {{Returning from 'initX'}}
                                // expected-note@-4 {{Object leaked: allocated object of type 'MyObj *' is not referenced later in this execution path and has a retain count of +1}}
  // initI is inlined because the allocation happens within initY
  id y = [[MyObj alloc] initY];
                                // expected-note@-1 {{Calling 'initY'}}
                                // expected-note@-2 {{Returning from 'initY'}}

  // initZ is not inlined
  id z = [[MyObj alloc] initZ]; // expected-warning {{Potential leak of an object}}
                                // expected-note@-1 {{Object leaked: object allocated and stored into 'y' is not referenced later in this execution path and has a retain count of +1}}

  [x release];
  [z release];
}
@end


void CFOverAutorelease() {
  CFTypeRef object = CFCreateSomething(); // expected-note{{Call to function 'CFCreateSomething' returns a Core Foundation object of type 'CFTypeRef' with a +1 retain count}}
  CFAutorelease(object); // expected-note{{Object autoreleased}}
  CFAutorelease(object); // expected-note{{Object autoreleased}}
  return; // expected-warning{{Object autoreleased too many times}} expected-note{{Object was autoreleased 2 times but the object has a +1 retain count}}
}

void CFAutoreleaseUnowned() {
  CFTypeRef object = CFGetSomething(); // expected-note{{Call to function 'CFGetSomething' returns a Core Foundation object of type 'CFTypeRef' with a +0 retain count}}
  CFAutorelease(object); // expected-note{{Object autoreleased}}
  return; // expected-warning{{Object autoreleased too many times}} expected-note{{Object was autoreleased but has a +0 retain count}}
}

void CFAutoreleaseUnownedMixed() {
  CFTypeRef object = CFGetSomething(); // expected-note{{Call to function 'CFGetSomething' returns a Core Foundation object of type 'CFTypeRef' with a +0 retain count}}
  CFAutorelease(object); // expected-note{{Object autoreleased}}
  [(id)object autorelease]; // expected-note{{Object autoreleased}}
  return; // expected-warning{{Object autoreleased too many times}} expected-note{{Object was autoreleased 2 times but the object has a +0 retain count}}
}

@interface PropertiesAndIvars : NSObject
@property (strong) id ownedProp;
@property (unsafe_unretained) id unownedProp;
@property (nonatomic, strong) id manualProp;
@end

@interface NSObject (PropertiesAndIvarsHelper)
- (void)myMethod;
@end

@implementation PropertiesAndIvars {
  id _ivarOnly;
}

- (id)manualProp {
  return _manualProp;
}

- (void)testOverreleaseUnownedIvar {
  [_unownedProp retain]; // FIXME-note {{Object loaded from instance variable}}
  // FIXME-note@-1 {{Reference count incremented. The object now has a +1 retain count}}
  [_unownedProp release]; // FIXME-note {{Reference count decremented}}
  [_unownedProp release]; // FIXME-note {{Incorrect decrement of the reference count of an object that is not owned at this point by the caller}}
  // FIXME-warning@-1 {{not owned at this point by the caller}}
}

- (void)testOverreleaseOwnedIvarUse {
  [_ownedProp retain]; // FIXME-note {{Object loaded from instance variable}}
  // FIXME-note@-1 {{Reference count incremented. The object now has a +1 retain count}}
  [_ownedProp release]; // FIXME-note {{Reference count decremented}}
  [_ownedProp release]; // FIXME-note {{Strong instance variable relinquished. Object released}}
  [_ownedProp myMethod]; // FIXME-note {{Reference-counted object is used after it is released}}
  // FIXME-warning@-1 {{used after it is released}}
}

- (void)testOverreleaseIvarOnlyUse {
  [_ivarOnly retain]; // FIXME-note {{Object loaded from instance variable}}
  // FIXME-note@-1 {{Reference count incremented. The object now has a +1 retain count}}
  [_ivarOnly release]; // FIXME-note {{Reference count decremented}}
  [_ivarOnly release]; // FIXME-note {{Strong instance variable relinquished. Object released}}
  [_ivarOnly myMethod]; // FIXME-note {{Reference-counted object is used after it is released}}
  // FIXME-warning@-1 {{used after it is released}}
}

- (void)testOverreleaseOwnedIvarAutorelease {
  [_ownedProp retain]; // FIXME-note {{Object loaded from instance variable}}
  // FIXME-note@-1 {{Reference count incremented. The object now has a +1 retain count}}
  [_ownedProp release]; // FIXME-note {{Reference count decremented}}
  [_ownedProp autorelease]; // FIXME-note {{Object autoreleased}}
  [_ownedProp autorelease]; // FIXME-note {{Object autoreleased}}
  // FIXME-note@+1 {{Object was autoreleased 2 times but the object has a +0 retain count}}
} // FIXME-warning{{Object autoreleased too many times}}

- (void)testOverreleaseIvarOnlyAutorelease {
  [_ivarOnly retain]; // FIXME-note {{Object loaded from instance variable}}
  // FIXME-note@-1 {{Reference count incremented. The object now has a +1 retain count}}
  [_ivarOnly release]; // FIXME-note {{Reference count decremented}}
  [_ivarOnly autorelease]; // FIXME-note {{Object autoreleased}}
  [_ivarOnly autorelease]; // FIXME-note {{Object autoreleased}}
  // FIXME-note@+1 {{Object was autoreleased 2 times but the object has a +0 retain count}}
} // FIXME-warning{{Object autoreleased too many times}}

@end



