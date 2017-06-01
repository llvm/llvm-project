// RUN: %clang_analyze_cc1 -triple %itanium_abi_triple -analyzer-checker=core,osx.cocoa.RetainCount -analyzer-store=region -analyzer-max-loop 6 -verify %s

//===----------------------------------------------------------------------===//
// The following code is reduced using delta-debugging from
// Foundation.h (Mac OS X).
//
// It includes the basic definitions for the test cases below.
// Not including Foundation.h directly makes this test case both svelte and
// portable to non-Mac platforms.
//===----------------------------------------------------------------------===//

typedef unsigned int __darwin_natural_t;
typedef unsigned long UInt32;
typedef signed long CFIndex;
typedef const void * CFTypeRef;
typedef const struct __CFString * CFStringRef;
typedef const struct __CFAllocator * CFAllocatorRef;
extern const CFAllocatorRef kCFAllocatorDefault;
extern CFTypeRef CFRetain(CFTypeRef cf);
extern void CFRelease(CFTypeRef cf);
typedef struct {
}
CFArrayCallBacks;
extern const CFArrayCallBacks kCFTypeArrayCallBacks;
typedef const struct __CFArray * CFArrayRef;
typedef struct __CFArray * CFMutableArrayRef;
extern CFMutableArrayRef CFArrayCreateMutable(CFAllocatorRef allocator, CFIndex capacity, const CFArrayCallBacks *callBacks);
extern const void *CFArrayGetValueAtIndex(CFArrayRef theArray, CFIndex idx);
typedef const struct __CFDictionary * CFDictionaryRef;
typedef UInt32 CFStringEncoding;
enum {
kCFStringEncodingMacRoman = 0,     kCFStringEncodingWindowsLatin1 = 0x0500,     kCFStringEncodingISOLatin1 = 0x0201,     kCFStringEncodingNextStepLatin = 0x0B01,     kCFStringEncodingASCII = 0x0600,     kCFStringEncodingUnicode = 0x0100,     kCFStringEncodingUTF8 = 0x08000100,     kCFStringEncodingNonLossyASCII = 0x0BFF      ,     kCFStringEncodingUTF16 = 0x0100,     kCFStringEncodingUTF16BE = 0x10000100,     kCFStringEncodingUTF16LE = 0x14000100,      kCFStringEncodingUTF32 = 0x0c000100,     kCFStringEncodingUTF32BE = 0x18000100,     kCFStringEncodingUTF32LE = 0x1c000100  };
extern CFStringRef CFStringCreateWithCString(CFAllocatorRef alloc, const char *cStr, CFStringEncoding encoding);
typedef double CFTimeInterval;
typedef CFTimeInterval CFAbsoluteTime;
typedef const struct __CFDate * CFDateRef;
extern CFDateRef CFDateCreate(CFAllocatorRef allocator, CFAbsoluteTime at);
extern CFAbsoluteTime CFDateGetAbsoluteTime(CFDateRef theDate);
typedef __darwin_natural_t natural_t;
typedef natural_t mach_port_name_t;
typedef mach_port_name_t mach_port_t;
typedef signed char BOOL;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject
- (BOOL)isEqual:(id)object;
- (id)retain;
- (oneway void)release;
@end  @protocol NSCopying  - (id)copyWithZone:(NSZone *)zone;
@end  @protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@interface NSObject <NSObject> {}
- (id)init;
+ (id)allocWithZone:(NSZone *)zone;
+ (id)alloc;
- (void)dealloc;
@end
typedef float CGFloat;
typedef double NSTimeInterval;
@interface NSDate : NSObject <NSCopying, NSCoding>  - (NSTimeInterval)timeIntervalSinceReferenceDate;
@end      enum {
NSObjCNoType = 0,     NSObjCVoidType = 'v',     NSObjCCharType = 'c',     NSObjCShortType = 's',     NSObjCLongType = 'l',     NSObjCLonglongType = 'q',     NSObjCFloatType = 'f',     NSObjCDoubleType = 'd',      NSObjCBoolType = 'B',      NSObjCSelectorType = ':',     NSObjCObjectType = '@',     NSObjCStructType = '{',     NSObjCPointerType = '^',     NSObjCStringType = '*',     NSObjCArrayType = '[',     NSObjCUnionType = '(',     NSObjCBitfield = 'b' }
__attribute__((deprecated));
typedef int kern_return_t;
typedef kern_return_t mach_error_t;
typedef mach_port_t io_object_t;
typedef io_object_t io_service_t;
typedef struct __DASession * DASessionRef;
extern DASessionRef DASessionCreate( CFAllocatorRef allocator );
typedef struct __DADisk * DADiskRef;
extern DADiskRef DADiskCreateFromBSDName( CFAllocatorRef allocator, DASessionRef session, const char * name );
extern DADiskRef DADiskCreateFromIOMedia( CFAllocatorRef allocator, DASessionRef session, io_service_t media );
extern CFDictionaryRef DADiskCopyDescription( DADiskRef disk );
extern DADiskRef DADiskCopyWholeDisk( DADiskRef disk );
@interface NSAppleEventManager : NSObject {
}
@end enum {
kDAReturnSuccess = 0,     kDAReturnError = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x01,     kDAReturnBusy = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x02,     kDAReturnBadArgument = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x03,     kDAReturnExclusiveAccess = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x04,     kDAReturnNoResources = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x05,     kDAReturnNotFound = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x06,     kDAReturnNotMounted = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x07,     kDAReturnNotPermitted = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x08,     kDAReturnNotPrivileged = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x09,     kDAReturnNotReady = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x0A,     kDAReturnNotWritable = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x0B,     kDAReturnUnsupported = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x0C };
typedef mach_error_t DAReturn;
typedef const struct __DADissenter * DADissenterRef;
extern DADissenterRef DADissenterCreate( CFAllocatorRef allocator, DAReturn status, CFStringRef string );
@interface NSNumber : NSObject
- (id)initWithInt:(int)value;
@end
typedef unsigned long NSUInteger;
@interface NSArray : NSObject
-(id) initWithObjects:(const id *)objects count:(NSUInteger) cnt;
@end

//===----------------------------------------------------------------------===//
// Test cases.
//===----------------------------------------------------------------------===//

// Test to see if we *issue* an error when we store the pointer
// to a struct.  This differs from basic store.

CFAbsoluteTime CFAbsoluteTimeGetCurrent(void);

struct foo {
  NSDate* f;
};

// FIXME: We should be warning about a use-after-free here, but we
// temporarily "escape" retain counted objects stored to structs very eagerly
// until we can properly tell whether they have escaped via a return value
// or not.
CFAbsoluteTime f4() {
  struct foo x;
  
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t);  
  [((NSDate*) date) retain];
  CFRelease(date);
  CFDateGetAbsoluteTime(date); // no-warning
  x.f = (NSDate*) date;  
  [((NSDate*) date) release];
  // FIXME: the following line should warn.
  t = CFDateGetAbsoluteTime(date);   // no-warning
  return t;
}

// Test that assigning to an self.ivar loses track of an object.
// This is a temporary hack to reduce false positives.
@interface Test3 : NSObject {
  id myObj;
}
- (void)test_self_assign_ivar;
@end

@implementation Test3
- (void)test_self_assign_ivar {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t); // no-warning
  myObj = (id) date;
}
@end

//===------------------------------------------------------------------------------------------===//
// <rdar://problem/7257223> (also <rdar://problem/7283470>) - False positive due to not invalidating
//  the reference count of a tracked region that was itself invalidated.
//===------------------------------------------------------------------------------------------===//

typedef struct __rdar_7257223 { CFDateRef x; } RDar7257223;
void rdar_7257223_aux(RDar7257223 *p);

CFDateRef rdar7257223_Create(void) {
  RDar7257223 s;
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  s.x = CFDateCreate(0, t); // no-warning
  rdar_7257223_aux(&s);
  return s.x;
}

CFDateRef rdar7257223_Create_2(void) {
  RDar7257223 s;
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  s.x = CFDateCreate(0, t); // no-warning
  return s.x;
}

void rdar7283470(void) {
  NSNumber *numbers[] = {
    [[NSNumber alloc] initWithInt:1], // no-warning
    [[NSNumber alloc] initWithInt:2], // no-warning
    [[NSNumber alloc] initWithInt:3], // no-warning
    [[NSNumber alloc] initWithInt:4], // no-warning
    [[NSNumber alloc] initWithInt:5]  // no-warning
  };
  
  for (unsigned i = 0 ; i < sizeof(numbers) / sizeof(numbers[0]) ; ++i)
    [numbers[i] release];
}

void rdar7283470_positive(void) {
  NSNumber *numbers[] = {
    [[NSNumber alloc] initWithInt:1], // expected-warning{{leak}}
    [[NSNumber alloc] initWithInt:2], // expected-warning{{leak}}
    [[NSNumber alloc] initWithInt:3], // expected-warning{{leak}}
    [[NSNumber alloc] initWithInt:4], // expected-warning{{leak}}
    [[NSNumber alloc] initWithInt:5]  // expected-warning{{leak}} 
  };
}

void rdar7283470_2(void) {
  NSNumber *numbers[] = {
    [[NSNumber alloc] initWithInt:1], // no-warning
    [[NSNumber alloc] initWithInt:2], // no-warning
    [[NSNumber alloc] initWithInt:3], // no-warning
    [[NSNumber alloc] initWithInt:4], // no-warning
    [[NSNumber alloc] initWithInt:5]  // no-warning
  };
  
  NSArray *s_numbers =[[NSArray alloc] initWithObjects:&numbers[0] count:sizeof(numbers) / sizeof(numbers[0])];
  
  for (unsigned i = 0 ; i < sizeof(numbers) / sizeof(numbers[0]) ; ++i)
    [numbers[i] release];
  
  [s_numbers release];
}

void rdar7283470_2_positive(void) {
  NSNumber *numbers[] = {
    [[NSNumber alloc] initWithInt:1], // no-warning
    [[NSNumber alloc] initWithInt:2], // no-warning
    [[NSNumber alloc] initWithInt:3], // no-warning
    [[NSNumber alloc] initWithInt:4], // no-warning
    [[NSNumber alloc] initWithInt:5]  // no-warning
  };
  
  NSArray *s_numbers =[[NSArray alloc] initWithObjects: &numbers[0] count:sizeof(numbers) / sizeof(numbers[0])]; // expected-warning{{leak}}
  
  for (unsigned i = 0 ; i < sizeof(numbers) / sizeof(numbers[0]) ; ++i)
    [numbers[i] release];
}

void pr6699(int x) {
  CFDateRef values[2];
  values[0] = values[1] = 0;

  if (x) {
    CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
    values[1] = CFDateCreate(0, t);
  }

  if (values[1]) {
    // A bug in RegionStore::RemoveDeadBindings caused 'values[1]' to get prematurely
    // pruned from the store.
    CFRelease(values[1]); // no-warning
  }
}

// <rdar://problem/8261992> Idempotent operation checker false positive with ObjC ivars
@interface R8261992 : NSObject {
  @package int myIvar;
}
@end

static void R8261992_ChangeMyIvar(R8261992 *tc) {
    tc->myIvar = 5;
}

void R8261992_test(R8261992 *tc) {
  int temp = tc->myIvar;
  // The ivar binding for tc->myIvar gets invalidated.
  R8261992_ChangeMyIvar(tc);
  tc->myIvar = temp; // no-warning
  tc = [[R8261992 alloc] init];
  temp = tc->myIvar; // no-warning
  // The ivar binding for tc->myIvar gets invalidated.
  R8261992_ChangeMyIvar(tc);
  tc->myIvar = temp;
  [tc release]; // no-warning
  // did we analyze this?
  int *p = 0x0;
  *p = 0xDEADBEEF; // expected-warning{{null}}
}

