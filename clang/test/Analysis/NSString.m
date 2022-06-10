// RUN: %clang_analyze_cc1 -triple i386-apple-darwin10 -analyzer-checker=core,osx.cocoa.NilArg,osx.cocoa.RetainCount,alpha.core -verify -Wno-objc-root-class %s
// RUN: %clang_analyze_cc1 -triple i386-apple-darwin10 -analyzer-checker=core,osx.cocoa.NilArg,osx.cocoa.RetainCount,alpha.core -analyzer-config mode=shallow -verify -Wno-objc-root-class %s
// RUN: %clang_analyze_cc1 -DTEST_64 -triple x86_64-apple-darwin10 -analyzer-checker=core,osx.cocoa.NilArg,osx.cocoa.RetainCount,alpha.core -verify -Wno-objc-root-class %s
// RUN: %clang_analyze_cc1 -DOSATOMIC_USE_INLINED -triple i386-apple-darwin10 -analyzer-checker=core,osx.cocoa.NilArg,osx.cocoa.RetainCount,alpha.core -verify -Wno-objc-root-class %s

//===----------------------------------------------------------------------===//
// The following code is reduced using delta-debugging from
// Foundation.h (Mac OS X).
//
// It includes the basic definitions for the test cases below.
// Not directly including Foundation.h directly makes this test case 
// both svelte and portable to non-Mac platforms.
//===----------------------------------------------------------------------===//

#ifdef TEST_64
typedef long long int64_t;
_Bool OSAtomicCompareAndSwap64Barrier( int64_t __oldValue, int64_t __newValue, volatile int64_t *__theValue );
#define COMPARE_SWAP_BARRIER OSAtomicCompareAndSwap64Barrier
typedef int64_t intptr_t;
#else
typedef int int32_t;
_Bool OSAtomicCompareAndSwap32Barrier( int32_t __oldValue, int32_t __newValue, volatile int32_t *__theValue );
#define COMPARE_SWAP_BARRIER OSAtomicCompareAndSwap32Barrier
typedef int32_t intptr_t;
#endif

typedef const void * CFTypeRef;
typedef const struct __CFString * CFStringRef;
typedef const struct __CFAllocator * CFAllocatorRef;
extern const CFAllocatorRef kCFAllocatorDefault;
extern CFTypeRef CFRetain(CFTypeRef cf);
void CFRelease(CFTypeRef cf);
typedef const struct __CFDictionary * CFDictionaryRef;
const void *CFDictionaryGetValue(CFDictionaryRef theDict, const void *key);
extern CFStringRef CFStringCreateWithFormat(CFAllocatorRef alloc, CFDictionaryRef formatOptions, CFStringRef format, ...);
typedef signed char BOOL;
typedef int NSInteger;
typedef unsigned int NSUInteger;
@class NSString, Protocol;
extern void NSLog(NSString *format, ...) __attribute__((format(__NSString__, 1, 2)));
typedef NSInteger NSComparisonResult;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject
- (BOOL)isEqual:(id)object;
- (oneway void)release;
- (id)retain;
- (id)autorelease;
@end
@protocol NSCopying
- (id)copyWithZone:(NSZone *)zone;
@end
@protocol NSMutableCopying
- (id)mutableCopyWithZone:(NSZone *)zone;
@end
@protocol NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@interface NSObject <NSObject> {}
- (id)init;
+ (id)alloc;
@end
extern id NSAllocateObject(Class aClass, NSUInteger extraBytes, NSZone *zone);
typedef struct {} NSFastEnumerationState;
@protocol NSFastEnumeration
- (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id *)stackbuf count:(NSUInteger)len;
@end
@class NSString;
typedef struct _NSRange {} NSRange;
@interface NSArray : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>
- (NSUInteger)count;
@end
@interface NSMutableArray : NSArray
- (void)addObject:(id)anObject;
- (id)initWithCapacity:(NSUInteger)numItems;
@end
typedef unsigned short unichar;
@class NSData, NSArray, NSDictionary, NSCharacterSet, NSData, NSURL, NSError, NSLocale;
typedef NSUInteger NSStringCompareOptions;
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>    - (NSUInteger)length;
- (NSComparisonResult)compare:(NSString *)string;
- (NSComparisonResult)compare:(NSString *)string options:(NSStringCompareOptions)mask;
- (NSComparisonResult)compare:(NSString *)string options:(NSStringCompareOptions)mask range:(NSRange)compareRange;
- (NSComparisonResult)compare:(NSString *)string options:(NSStringCompareOptions)mask range:(NSRange)compareRange locale:(id)locale;
- (NSComparisonResult)caseInsensitiveCompare:(NSString *)string;
- (NSArray *)componentsSeparatedByCharactersInSet:(NSCharacterSet *)separator;
+ (id)stringWithFormat:(NSString *)format, ... __attribute__((format(__NSString__, 1, 2)));
@end
@interface NSSimpleCString : NSString {} @end
@interface NSConstantString : NSSimpleCString @end
extern void *_NSConstantStringClassReference;

//===----------------------------------------------------------------------===//
// Test cases.
//===----------------------------------------------------------------------===//

NSComparisonResult f1(NSString* s) {
  NSString *aString = 0;
  return [s compare:aString]; // expected-warning {{Argument to 'NSString' method 'compare:' cannot be nil}}
}

NSComparisonResult f2(NSString* s) {
  NSString *aString = 0;
  return [s caseInsensitiveCompare:aString]; // expected-warning {{Argument to 'NSString' method 'caseInsensitiveCompare:' cannot be nil}}
}

NSComparisonResult f3(NSString* s, NSStringCompareOptions op) {
  NSString *aString = 0;
  return [s compare:aString options:op]; // expected-warning {{Argument to 'NSString' method 'compare:options:' cannot be nil}}
}

NSComparisonResult f4(NSString* s, NSStringCompareOptions op, NSRange R) {
  NSString *aString = 0;
  return [s compare:aString options:op range:R]; // expected-warning {{Argument to 'NSString' method 'compare:options:range:' cannot be nil}}
}

NSComparisonResult f5(NSString* s, NSStringCompareOptions op, NSRange R) {
  NSString *aString = 0;
  return [s compare:aString options:op range:R locale:0]; // expected-warning {{Argument to 'NSString' method 'compare:options:range:locale:' cannot be nil}}
}

NSArray *f6(NSString* s) {
  return [s componentsSeparatedByCharactersInSet:0]; // expected-warning {{Argument to 'NSString' method 'componentsSeparatedByCharactersInSet:' cannot be nil}}
}

NSString* f7(NSString* s1, NSString* s2, NSString* s3) {

  NSString* s4 = (NSString*)
    CFStringCreateWithFormat(kCFAllocatorDefault, 0,  // expected-warning{{leak}}
                             (CFStringRef) __builtin___CFStringMakeConstantString("%@ %@ (%@)"), 
                             s1, s2, s3);

  CFRetain(s4);
  return s4;
}

NSMutableArray* f8(void) {
  
  NSString* s = [[NSString alloc] init];
  NSMutableArray* a = [[NSMutableArray alloc] initWithCapacity:2];
  [a addObject:s];
  [s release]; // no-warning
  return a;
}

void f9(void) {
  
  NSString* s = [[NSString alloc] init];
  NSString* q = s;
  [s release];
  [q release]; // expected-warning {{used after it is released}}
}

NSString* f10(void) {
  static NSString* s = 0;
  if (!s) s = [[NSString alloc] init];
  return s; // no-warning
}

// Test case for regression reported in <rdar://problem/6452745>.
// Essentially 's' should not be considered allocated on the false branch.
// This exercises the 'EvalAssume' logic in GRTransferFuncs (CFRefCount.cpp).
NSString* f11(CFDictionaryRef dict, const char* key) {
  NSString* s = (NSString*) CFDictionaryGetValue(dict, key);
  [s retain];
  if (s) {
    [s release];
  }
  return 0;
}

// Test case for passing a tracked object by-reference to a function we
// don't understand.
void unknown_function_f12(NSString** s);
void f12(void) {
  NSString *string = [[NSString alloc] init];
  unknown_function_f12(&string); // no-warning
}

// Test double release of CFString (PR 4014).
void f13(void) {
  CFStringRef ref = CFStringCreateWithFormat(kCFAllocatorDefault, ((void*)0), ((CFStringRef) __builtin___CFStringMakeConstantString ("" "%d" "")), 100);
  CFRelease(ref);
  CFRelease(ref); // expected-warning{{Reference-counted object is used after it is released}}
}

@interface MyString : NSString
@end

void f14(MyString *s) {
  [s compare:0]; // expected-warning {{Argument to 'MyString' method 'compare:' cannot be nil}}
}

// Test regular use of -autorelease
@interface TestAutorelease
-(NSString*) getString;
@end
@implementation TestAutorelease
-(NSString*) getString {
  NSString *str = [[NSString alloc] init];
  return [str autorelease]; // no-warning
}
- (void)m1
{
 NSString *s = [[NSString alloc] init]; // expected-warning{{leak}}
 [s retain];
 [s autorelease];
}
- (void)m2
{
 NSString *s = [[[NSString alloc] init] autorelease]; // expected-warning{{leak}}
 [s retain];
}
- (void)m3
{
 NSString *s = [[[NSString alloc] init] autorelease];
 [s retain];
 [s autorelease];
}
- (void)m4
{
 NSString *s = [[NSString alloc] init]; // expected-warning{{leak}}
 [s retain];
}
- (void)m5
{
 NSString *s = [[NSString alloc] init];
 [s autorelease];
}
@end

@interface C1 : NSObject {}
- (NSString*) getShared;
+ (C1*) sharedInstance;
@end
@implementation C1 : NSObject {}
- (NSString*) getShared {
  static NSString* s = 0;
  if (!s) s = [[NSString alloc] init];    
  return s; // no-warning  
}
+ (C1 *)sharedInstance {
  static C1 *sharedInstance = 0;
  if (!sharedInstance) {
    sharedInstance = [[C1 alloc] init];
  }
  return sharedInstance; // no-warning
}
@end

@interface SharedClass : NSObject
+ (id)sharedInstance;
- (id)notShared;
@end

@implementation SharedClass

- (id)_init {
    if ((self = [super init])) {
        NSLog(@"Bar");
    }
    return self;
}

- (id)notShared {
  return [[SharedClass alloc] _init]; // expected-warning{{leak}}
}

+ (id)sharedInstance {
    static SharedClass *_sharedInstance = 0;
    if (!_sharedInstance) {
        _sharedInstance = [[SharedClass alloc] _init];
    }
    return _sharedInstance; // no-warning
}
@end

id testSharedClassFromFunction(void) {
  return [[SharedClass alloc] _init]; // no-warning
}

#if !(defined(OSATOMIC_USE_INLINED) && OSATOMIC_USE_INLINED)
// Test OSCompareAndSwap
_Bool OSAtomicCompareAndSwapPtr( void *__oldValue, void *__newValue, void * volatile *__theValue );
extern BOOL objc_atomicCompareAndSwapPtr(id predicate, id replacement, volatile id *objectLocation);
#else
// Test that the body farm models are still used even when a body is available.
_Bool opaque_OSAtomicCompareAndSwapPtr( void *__oldValue, void *__newValue, void * volatile *__theValue );
_Bool OSAtomicCompareAndSwapPtr( void *__oldValue, void *__newValue, void * volatile *__theValue ) {
  return opaque_OSAtomicCompareAndSwapPtr(__oldValue, __newValue, __theValue);
}
// Test that the analyzer doesn't crash when the farm model is used. 
// The analyzer ignores the autosynthesized code.
_Bool OSAtomicCompareAndSwapEmptyFunction( void *__oldValue, void *__newValue, void * volatile *__theValue ) {
  return 0;
}
extern BOOL opaque_objc_atomicCompareAndSwapPtr(id predicate, id replacement, volatile id *objectLocation);
extern BOOL objc_atomicCompareAndSwapPtr(id predicate, id replacement, volatile id *objectLocation) {
  return opaque_objc_atomicCompareAndSwapPtr(predicate, replacement, objectLocation);
}
#endif

void testOSCompareAndSwap(void) {
  NSString *old = 0;
  NSString *s = [[NSString alloc] init]; // no-warning
  if (!OSAtomicCompareAndSwapPtr(0, s, (void**) &old))
    [s release];
  else    
    [old release];
}

void testOSCompareAndSwapXXBarrier_local(void) {
  NSString *old = 0;
  NSString *s = [[NSString alloc] init]; // no-warning
  if (!COMPARE_SWAP_BARRIER((intptr_t) 0, (intptr_t) s, (intptr_t*) &old))
    [s release];
  else    
    [old release];
}

void testOSCompareAndSwapXXBarrier_local_no_direct_release(void) {
  NSString *old = 0;
  NSString *s = [[NSString alloc] init]; // no-warning
  if (!COMPARE_SWAP_BARRIER((intptr_t) 0, (intptr_t) s, (intptr_t*) &old))
    return;
  else    
    [old release];
}

int testOSCompareAndSwapXXBarrier_id(Class myclass, id xclass) {
  if (COMPARE_SWAP_BARRIER(0, (intptr_t) myclass, (intptr_t*) &xclass))
    return 1;
  return 0;
}

void test_objc_atomicCompareAndSwap_local(void) {
  NSString *old = 0;
  NSString *s = [[NSString alloc] init]; // no-warning
  if (!objc_atomicCompareAndSwapPtr(0, s, &old))
    [s release];
  else    
    [old release];
}

void test_objc_atomicCompareAndSwap_local_no_direct_release(void) {
  NSString *old = 0;
  NSString *s = [[NSString alloc] init]; // no-warning
  if (!objc_atomicCompareAndSwapPtr(0, s, &old))
    return;
  else    
    [old release];
}

void test_objc_atomicCompareAndSwap_parameter(NSString **old) {
  NSString *s = [[NSString alloc] init]; // no-warning
  if (!objc_atomicCompareAndSwapPtr(0, s, old))
    [s release];
  else    
    [*old release];
}

void test_objc_atomicCompareAndSwap_parameter_no_direct_release(NSString **old) {
  NSString *s = [[NSString alloc] init]; // expected-warning{{leak}}
  if (!objc_atomicCompareAndSwapPtr(0, s, old))
    return;
  else    
    [*old release];
}


// Test stringWithFormat (<rdar://problem/6815234>)
void test_stringWithFormat(void) {  
  NSString *string = [[NSString stringWithFormat:@"%ld", (long) 100] retain];
  [string release];
  [string release]; // expected-warning{{Incorrect decrement of the reference count}}
}

// Test isTrackedObjectType(void).
typedef NSString* WonkyTypedef;
@interface TestIsTracked
+ (WonkyTypedef)newString;
@end

void test_isTrackedObjectType(void) {
  NSString *str = [TestIsTracked newString]; // expected-warning{{Potential leak}}
}

// Test isTrackedCFObjectType().
@interface TestIsCFTracked
+ (CFStringRef) badNewCFString;
+ (CFStringRef) newCFString;
@end

@implementation TestIsCFTracked
+ (CFStringRef) newCFString {
  return CFStringCreateWithFormat(kCFAllocatorDefault, ((void*)0), ((CFStringRef) __builtin___CFStringMakeConstantString ("" "%d" "")), 100); // no-warning
}
+ (CFStringRef) badNewCFString {
  return CFStringCreateWithFormat(kCFAllocatorDefault, ((void*)0), ((CFStringRef) __builtin___CFStringMakeConstantString ("" "%d" "")), 100); // expected-warning{{leak}}
}

// Test @synchronized
void test_synchronized(id x) {
  @synchronized(x) {
    NSString *string = [[NSString stringWithFormat:@"%ld", (long) 100] retain]; // expected-warning {{leak}}
  }
}
@end

void testOSCompareAndSwapXXBarrier_parameter(NSString **old) {
  NSString *s = [[NSString alloc] init]; // no-warning
  if (!COMPARE_SWAP_BARRIER((intptr_t) 0, (intptr_t) s, (intptr_t*) old))
    [s release];
  else    
    [*old release];
}

void testOSCompareAndSwapXXBarrier_parameter_no_direct_release(NSString **old) {
  NSString *s = [[NSString alloc] init]; // no-warning
  if (!COMPARE_SWAP_BARRIER((intptr_t) 0, (intptr_t) s, (intptr_t*) old))
    [s release];
  else    
    return;
}

@interface AlwaysInlineBodyFarmBodies : NSObject {
  NSString *_value;
}
  - (NSString *)_value;
  - (void)callValue;
@end

@implementation AlwaysInlineBodyFarmBodies

- (NSString *)_value {
  if (!_value) {
    NSString *s = [[NSString alloc] init];
    if (!OSAtomicCompareAndSwapPtr(0, s, (void**)&_value)) {
      [s release];
    }
  }
  return _value;
}

- (void)callValue {
  [self _value];
}
@end
