// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core,osx.cocoa.RetainCount,deadcode -verify -fblocks -analyzer-opt-analyze-nested-blocks -fobjc-arc -analyzer-output=plist-multi-file -o %t.plist %s
// RUN: %normalize_plist <%t.plist | diff -ub %S/Inputs/expected-plists/objc-arc.m.plist -

typedef signed char BOOL;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
typedef unsigned long NSUInteger;

@protocol NSObject
- (BOOL)isEqual:(id)object;
@end
@protocol NSCopying
- (id)copyWithZone:(NSZone *)zone;
@end
@protocol NSCoding;
@protocol NSMutableCopying;
@protocol NSFastEnumeration
- (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone;
@end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@interface NSObject <NSObject> {}
+ (id)alloc;
- (id)init;
- (NSString *)description;
@end
@interface NSArray : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>
- (NSUInteger)count;
- (id)initWithObjects:(const id [])objects count:(NSUInteger)cnt;
+ (id)arrayWithObject:(id)anObject;
+ (id)arrayWithObjects:(const id [])objects count:(NSUInteger)cnt;
+ (id)arrayWithObjects:(id)firstObj, ... __attribute__((sentinel(0,1)));
- (id)initWithObjects:(id)firstObj, ... __attribute__((sentinel(0,1)));
- (id)initWithArray:(NSArray *)array;
@end

typedef const struct __CFAllocator * CFAllocatorRef;
extern const CFAllocatorRef kCFAllocatorDefault;
typedef double CFTimeInterval;
typedef CFTimeInterval CFAbsoluteTime;
extern CFAbsoluteTime CFAbsoluteTimeGetCurrent(void);
typedef const struct __CFDate * CFDateRef;
extern CFDateRef CFDateCreate(CFAllocatorRef allocator, CFAbsoluteTime at);

typedef const void* objc_objectptr_t;
__attribute__((ns_returns_retained)) id objc_retainedObject(objc_objectptr_t __attribute__((cf_consumed)) pointer);
__attribute__((ns_returns_not_retained)) id objc_unretainedObject(objc_objectptr_t pointer);

// Test the analyzer is working at all.
void test_working() {
  int *p = 0;
  *p = 0xDEADBEEF; // expected-warning {{null}}
}

// Test that in ARC mode that blocks are correctly automatically copied
// and not flagged as warnings by the analyzer.
typedef void (^Block)(void);
void testblock_bar(int x);

Block testblock_foo(int x) {
  Block b = ^{ testblock_bar(x); };
  return b; // no-warning
}

Block testblock_baz(int x) {
  return ^{ testblock_bar(x); }; // no-warning
}

Block global_block;

void testblock_qux(int x) {
  global_block = ^{ testblock_bar(x); }; // no-warning
}

// Test that Objective-C pointers are null initialized.
void test_nil_initialized() {
  id x;
  if (x == 0)
    return;
  int *p = 0;
  *p = 0xDEADBEEF; // no-warning
}

// Test that we don't flag leaks of Objective-C objects.
void test_alloc() {
  [NSObject alloc]; // no-warning
}

// Test that CF allocations are still caught as leaks.
void test_cf_leak() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t); // expected-warning {{Potential leak}}
  (void) date;
}

// Test that 'init' methods do not try to claim ownerhip of an *unowned* allocated object
// in ARC mode.
@interface RDar9424890_A :  NSObject
- (id)initWithCleaner:(int)pop mop:(NSString *)mop ;
- (RDar9424890_A *)rdar9424890:(NSString *)identifier;
@end
@interface RDar9424890_B : NSObject
@end
@implementation RDar9424890_B
- (RDar9424890_A *)obj:(RDar9424890_A *)obj {
  static NSString *WhizFiz = @"WhizFiz";
  RDar9424890_A *cell = [obj rdar9424890:WhizFiz];
  if (cell == ((void*)0)) {
    cell = [[RDar9424890_A alloc] initWithCleaner:0 mop:WhizFiz]; // no-warning
  }
  return cell;
}
@end

// Test that dead store checking works in the prescence of "cleanups" in the AST.
void rdar9424882() {
  id x = [NSObject alloc]; // expected-warning {{Value stored to 'x' during its initialization is never read}}
}

// Test 
typedef const void *CFTypeRef;
typedef const struct __CFString *CFStringRef;

@interface NSString : NSObject
- (id) self;
@end

CFTypeRef CFCreateSomething();
CFStringRef CFCreateString();
CFTypeRef CFGetSomething();
CFStringRef CFGetString();

id CreateSomething();
NSString *CreateNSString();

void from_cf() {
  id obj1 = (__bridge_transfer id)CFCreateSomething(); // expected-warning{{never read}}
  id obj2 = (__bridge_transfer NSString*)CFCreateString();
  [obj2 self]; // Add a use, to show we can use the object after it has been transferred.
  id obj3 = (__bridge id)CFGetSomething();
  [obj3 self]; // Add a use, to show we can use the object after it has been bridged.
  id obj4 = (__bridge NSString*)CFGetString(); // expected-warning{{never read}}
  id obj5 = (__bridge id)CFCreateSomething(); // expected-warning{{never read}} expected-warning{{leak}}
  id obj6 = (__bridge NSString*)CFCreateString(); // expected-warning{{never read}} expected-warning{{leak}}
}

void to_cf(id obj) {
  CFTypeRef cf1 = (__bridge_retained CFTypeRef)CreateSomething(); // expected-warning{{never read}}
  CFStringRef cf2 = (__bridge_retained CFStringRef)CreateNSString(); // expected-warning{{never read}}
  CFTypeRef cf3 = (__bridge CFTypeRef)CreateSomething(); // expected-warning{{never read}}
  CFStringRef cf4 = (__bridge CFStringRef)CreateNSString();  // expected-warning{{never read}}
}

void test_objc_retainedObject() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t);
  id x = objc_retainedObject(date);
  (void) x;
}

void test_objc_unretainedObject() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t);  // expected-warning {{Potential leak}}
  id x = objc_unretainedObject(date);
  (void) x;
}

// Previously this resulted in a "return of stack address" warning.
id test_return() {
  id x = (__bridge_transfer id) CFCreateString();
  return x; // no-warning
}

void test_objc_arrays() {
    { // CASE ONE -- OBJECT IN ARRAY CREATED DIRECTLY
        NSObject *o = [[NSObject alloc] init];
        NSArray *a = [[NSArray alloc] initWithObjects:o, (void*)0];
        [a description];
        [o description];
    }

    { // CASE TWO -- OBJECT IN ARRAY CREATED BY DUPING AUTORELEASED ARRAY
        NSObject *o = [[NSObject alloc] init];
        NSArray *a1 = [NSArray arrayWithObjects:o, (void*)0];
        NSArray *a2 = [[NSArray alloc] initWithArray:a1];
        [a2 description];
        [o description];
    }

    { // CASE THREE -- OBJECT IN RETAINED @[]
        NSObject *o = [[NSObject alloc] init];
        NSArray *a3 = @[o];
        [a3 description];
        [o description];
    }
    {
      // CASE 4, verify analyzer still working.
      CFCreateString(); // expected-warning {{leak}}
    }
}

// <rdar://problem/11059275> - dispatch_set_context and ARC.
__attribute__((cf_returns_retained)) CFTypeRef CFBridgingRetain(id X);
typedef void* dispatch_object_t;
void dispatch_set_context(dispatch_object_t object, const void *context);

void rdar11059275(dispatch_object_t object) {
  NSObject *o = [[NSObject alloc] init];
  dispatch_set_context(object, CFBridgingRetain(o)); // no-warning  
}
void rdar11059275_positive() {
  NSObject *o = [[NSObject alloc] init]; // expected-warning {{leak}}
  CFBridgingRetain(o);
}
void rdar11059275_negative() {
  NSObject *o = [[NSObject alloc] init]; // no-warning
  (void) o;
}

__attribute__((ns_returns_retained)) id rdar14061675_helper() {
  return [[NSObject alloc] init];
}

id rdar14061675() {
  // ARC produces an implicit cast here. We need to make sure the combination
  // of that and the inlined call don't produce a spurious edge cycle.
  id result = rdar14061675_helper();
  *(volatile int *)0 = 1; // expected-warning{{Dereference of null pointer}}
  return result;
}

typedef const void * CFTypeRef;
typedef const struct __CFString * CFStringRef;
typedef const struct __CFAllocator * CFAllocatorRef;
extern const CFAllocatorRef kCFAllocatorDefault;

extern CFTypeRef CFRetain(CFTypeRef cf);
extern void CFRelease(CFTypeRef cf);


void check_bridge_retained_cast() {
    NSString *nsStr = [[NSString alloc] init];
    CFStringRef cfStr = (__bridge_retained CFStringRef)nsStr;
    CFRelease(cfStr); // no-warning
}

@interface A;
@end

void check_bridge_to_non_cocoa(CFStringRef s) {
  A *a = (__bridge_transfer A *) s; // no-crash
}

struct B;

struct B * check_bridge_to_non_cf() {
  NSString *s = [[NSString alloc] init];
  return (__bridge struct B*) s;
}
