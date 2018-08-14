// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core,osx.coreFoundation.CFRetainRelease,osx.cocoa.ClassRelease,osx.cocoa.RetainCount -analyzer-store=region -fobjc-gc-only -analyzer-output=text -verify %s
// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core,osx.coreFoundation.CFRetainRelease,osx.cocoa.ClassRelease,osx.cocoa.RetainCount -analyzer-store=region -fobjc-gc-only -analyzer-output=plist-multi-file %s -o %t.plist
// RUN: tail -n +11 %t.plist | diff -u -w - %S/Inputs/expected-plists/retain-release-path-notes-gc.m.plist

/***
This file is for testing the path-sensitive notes for retain/release errors.
Its goal is to have simple branch coverage of any path-based diagnostics,
not to actually check all possible retain/release errors.

This file is for notes that only appear in a GC-enabled analysis. 
Non-specific and ref-count-only notes should go in retain-release-path-notes.m.
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
@end

typedef struct CFType *CFTypeRef;
CFTypeRef CFRetain(CFTypeRef);
void CFRelease(CFTypeRef);

id NSMakeCollectable(CFTypeRef);
CFTypeRef CFMakeCollectable(CFTypeRef);

CFTypeRef CFCreateSomething();
CFTypeRef CFGetSomething();


void creationViaCFCreate () {
  CFTypeRef leaked = CFCreateSomething(); // expected-note{{Call to function 'CFCreateSomething' returns a Core Foundation object of type CFTypeRef with a +1 retain count.  Core Foundation objects are not automatically garbage collected}}
  return; // expected-warning{{leak}} expected-note{{Object leaked: object allocated and stored into 'leaked' is not referenced later in this execution path and has a retain count of +1}}
}

void makeCollectable () {
  CFTypeRef leaked = CFCreateSomething(); // expected-note{{Call to function 'CFCreateSomething' returns a Core Foundation object of type CFTypeRef with a +1 retain count.  Core Foundation objects are not automatically garbage collected}}
  CFRetain(leaked); // expected-note{{Reference count incremented. The object now has a +2 retain count}}
  CFMakeCollectable(leaked); // expected-note{{In GC mode a call to 'CFMakeCollectable' decrements an object's retain count and registers the object with the garbage collector. An object must have a 0 retain count to be garbage collected. After this call its retain count is +1}}
  NSMakeCollectable(leaked); // expected-note{{In GC mode a call to 'NSMakeCollectable' decrements an object's retain count and registers the object with the garbage collector. Since it now has a 0 retain count the object can be automatically collected by the garbage collector}}
  CFRetain(leaked); // expected-note{{Reference count incremented. The object now has a +1 retain count. The object is not eligible for garbage collection until the retain count reaches 0 again}}
  return; // expected-warning{{leak}} expected-note{{Object leaked: object allocated and stored into 'leaked' is not referenced later in this execution path and has a retain count of +1}}
}

void retainReleaseIgnored () {
  id object = [[NSObject alloc] init]; // expected-note{{Method returns an instance of NSObject with a +0 retain count}}
  [object retain]; // expected-note{{In GC mode the 'retain' message has no effect}}
  [object release]; // expected-note{{In GC mode the 'release' message has no effect}}
  [object autorelease]; // expected-note{{In GC mode an 'autorelease' has no effect}}
  CFRelease((CFTypeRef)object); // expected-warning{{Incorrect decrement of the reference count of an object that is not owned at this point by the caller}} expected-note{{Incorrect decrement of the reference count of an object that is not owned at this point by the caller}}
}

@implementation Foo (FundamentalRuleUnderGC)
- (id)getViolation {
  id object = (id) CFCreateSomething(); // expected-note{{Call to function 'CFCreateSomething' returns a Core Foundation object of type CFTypeRef with a +1 retain count.  Core Foundation objects are not automatically garbage collected}}
  return object; // expected-warning{{leak}} expected-note{{Object returned to caller as an owning reference (single retain count transferred to caller)}} expected-note{{Object leaked: object allocated and stored into 'object' and returned from method 'getViolation' is potentially leaked when using garbage collection.  Callers of this method do not expect a returned object with a +1 retain count since they expect the object to be managed by the garbage collector}}
}

- (id)copyViolation {
  id object = (id) CFCreateSomething(); // expected-note{{Call to function 'CFCreateSomething' returns a Core Foundation object of type CFTypeRef with a +1 retain count.  Core Foundation objects are not automatically garbage collected}}
  return object; // expected-warning{{leak}} expected-note{{Object returned to caller as an owning reference (single retain count transferred to caller)}} expected-note{{Object leaked: object allocated and stored into 'object' and returned from method 'copyViolation' is potentially leaked when using garbage collection.  Callers of this method do not expect a returned object with a +1 retain count since they expect the object to be managed by the garbage collector}}
}
@end

