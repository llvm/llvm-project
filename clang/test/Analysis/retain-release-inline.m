// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core,osx.coreFoundation.CFRetainRelease,osx.cocoa.ClassRelease,osx.cocoa.RetainCount -fblocks -verify %s

//===----------------------------------------------------------------------===//
// The following code is reduced using delta-debugging from Mac OS X headers:
//
// #include <Cocoa/Cocoa.h>
// #include <CoreFoundation/CoreFoundation.h>
// #include <DiskArbitration/DiskArbitration.h>
// #include <QuartzCore/QuartzCore.h>
// #include <Quartz/Quartz.h>
// #include <IOKit/IOKitLib.h>
//
// It includes the basic definitions for the test cases below.
//===----------------------------------------------------------------------===//
#define NULL 0
#define nil ((id)0)
typedef unsigned int __darwin_natural_t;
typedef unsigned long uintptr_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef unsigned int UInt32;
typedef signed long CFIndex;
typedef CFIndex CFByteOrder;
typedef struct {
  CFIndex location;
  CFIndex length;
} CFRange;
static __inline__ __attribute__((always_inline)) CFRange CFRangeMake(CFIndex loc, CFIndex len) {
  CFRange range;
  range.location = loc;
  range.length = len;
  return range;
}
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
extern void CFArrayAppendValue(CFMutableArrayRef theArray, const void *value);
typedef struct {
}
CFDictionaryKeyCallBacks;
extern const CFDictionaryKeyCallBacks kCFTypeDictionaryKeyCallBacks;
typedef struct {
}
CFDictionaryValueCallBacks;
extern const CFDictionaryValueCallBacks kCFTypeDictionaryValueCallBacks;
typedef const struct __CFDictionary * CFDictionaryRef;
typedef struct __CFDictionary * CFMutableDictionaryRef;
extern CFMutableDictionaryRef CFDictionaryCreateMutable(CFAllocatorRef allocator, CFIndex capacity, const CFDictionaryKeyCallBacks *keyCallBacks, const CFDictionaryValueCallBacks *valueCallBacks);
typedef UInt32 CFStringEncoding;
enum {
kCFStringEncodingMacRoman = 0,     kCFStringEncodingWindowsLatin1 = 0x0500,     kCFStringEncodingISOLatin1 = 0x0201,     kCFStringEncodingNextStepLatin = 0x0B01,     kCFStringEncodingASCII = 0x0600,     kCFStringEncodingUnicode = 0x0100,     kCFStringEncodingUTF8 = 0x08000100,     kCFStringEncodingNonLossyASCII = 0x0BFF      ,     kCFStringEncodingUTF16 = 0x0100,     kCFStringEncodingUTF16BE = 0x10000100,     kCFStringEncodingUTF16LE = 0x14000100,      kCFStringEncodingUTF32 = 0x0c000100,     kCFStringEncodingUTF32BE = 0x18000100,     kCFStringEncodingUTF32LE = 0x1c000100  };
extern CFStringRef CFStringCreateWithCString(CFAllocatorRef alloc, const char *cStr, CFStringEncoding encoding);
typedef double CFTimeInterval;
typedef CFTimeInterval CFAbsoluteTime;
extern CFAbsoluteTime CFAbsoluteTimeGetCurrent(void);
typedef const struct __CFDate * CFDateRef;
extern CFDateRef CFDateCreate(CFAllocatorRef allocator, CFAbsoluteTime at);
extern CFAbsoluteTime CFDateGetAbsoluteTime(CFDateRef theDate);
typedef __darwin_natural_t natural_t;
typedef natural_t mach_port_name_t;
typedef mach_port_name_t mach_port_t;
typedef int kern_return_t;
typedef kern_return_t mach_error_t;
enum {
kCFNumberSInt8Type = 1,     kCFNumberSInt16Type = 2,     kCFNumberSInt32Type = 3,     kCFNumberSInt64Type = 4,     kCFNumberFloat32Type = 5,     kCFNumberFloat64Type = 6,      kCFNumberCharType = 7,     kCFNumberShortType = 8,     kCFNumberIntType = 9,     kCFNumberLongType = 10,     kCFNumberLongLongType = 11,     kCFNumberFloatType = 12,     kCFNumberDoubleType = 13,      kCFNumberCFIndexType = 14,      kCFNumberNSIntegerType = 15,     kCFNumberCGFloatType = 16,     kCFNumberMaxType = 16    };
typedef CFIndex CFNumberType;
typedef const struct __CFNumber * CFNumberRef;
extern CFNumberRef CFNumberCreate(CFAllocatorRef allocator, CFNumberType theType, const void *valuePtr);
typedef const struct __CFAttributedString *CFAttributedStringRef;
typedef struct __CFAttributedString *CFMutableAttributedStringRef;
extern CFAttributedStringRef CFAttributedStringCreate(CFAllocatorRef alloc, CFStringRef str, CFDictionaryRef attributes) ;
extern CFMutableAttributedStringRef CFAttributedStringCreateMutableCopy(CFAllocatorRef alloc, CFIndex maxLength, CFAttributedStringRef aStr) ;
extern void CFAttributedStringSetAttribute(CFMutableAttributedStringRef aStr, CFRange range, CFStringRef attrName, CFTypeRef value) ;
typedef signed char BOOL;
typedef unsigned long NSUInteger;
@class NSString, Protocol;
extern void NSLog(NSString *format, ...) __attribute__((format(__NSString__, 1, 2)));
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject
- (BOOL)isEqual:(id)object;
- (id)retain;
- (oneway void)release;
- (Class)class;
- (id)autorelease;
- (id)init;
@end  @protocol NSCopying  - (id)copyWithZone:(NSZone *)zone;
@end  @protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone;
@end  @protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@interface NSObject <NSObject> {}
+ (id)allocWithZone:(NSZone *)zone;
+ (id)alloc;
+ (Class)class;
- (void)dealloc;
@end
@interface NSObject (NSCoderMethods)
- (id)awakeAfterUsingCoder:(NSCoder *)aDecoder;
@end
extern id NSAllocateObject(Class aClass, NSUInteger extraBytes, NSZone *zone);
typedef struct {
}
NSFastEnumerationState;
@protocol NSFastEnumeration  - (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id *)stackbuf count:(NSUInteger)len;
@end           @class NSString, NSDictionary;
@interface NSValue : NSObject <NSCopying, NSCoding>  - (void)getValue:(void *)value;
@end  @interface NSNumber : NSValue  - (char)charValue;
- (id)initWithInt:(int)value;
@end   @class NSString;
@interface NSArray : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>  - (NSUInteger)count;
@end  @interface NSArray (NSArrayCreation)  + (id)array;
@end       @interface NSAutoreleasePool : NSObject {
}
- (void)drain;
@end extern NSString * const NSBundleDidLoadNotification;
typedef double NSTimeInterval;
@interface NSDate : NSObject <NSCopying, NSCoding>  - (NSTimeInterval)timeIntervalSinceReferenceDate;
@end            typedef unsigned short unichar;
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>
- (NSUInteger)length;
- (NSString *)stringByAppendingString:(NSString *)aString;
- ( const char *)UTF8String;
- (id)initWithUTF8String:(const char *)nullTerminatedCString;
+ (id)stringWithUTF8String:(const char *)nullTerminatedCString;
@end        @class NSString, NSURL, NSError;
@interface NSData : NSObject <NSCopying, NSMutableCopying, NSCoding>  - (NSUInteger)length;
+ (id)dataWithBytesNoCopy:(void *)bytes length:(NSUInteger)length;
+ (id)dataWithBytesNoCopy:(void *)bytes length:(NSUInteger)length freeWhenDone:(BOOL)b;
@end   @class NSLocale, NSDate, NSCalendar, NSTimeZone, NSError, NSArray, NSMutableDictionary;
@interface NSDictionary : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>  - (NSUInteger)count;
@end    @interface NSMutableDictionary : NSDictionary  - (void)removeObjectForKey:(id)aKey;
- (void)setObject:(id)anObject forKey:(id)aKey;
@end  @interface NSMutableDictionary (NSMutableDictionaryCreation)  + (id)dictionaryWithCapacity:(NSUInteger)numItems;
@end  typedef double CGFloat;
struct CGSize {
};
typedef struct CGSize CGSize;
struct CGRect {
};
typedef struct CGRect CGRect;
typedef mach_port_t io_object_t;
typedef char io_name_t[128];
typedef io_object_t io_iterator_t;
typedef io_object_t io_service_t;
typedef struct IONotificationPort * IONotificationPortRef;
typedef void (*IOServiceMatchingCallback)(  void * refcon,  io_iterator_t iterator );
io_service_t IOServiceGetMatchingService(  mach_port_t mainPort,  CFDictionaryRef matching );
kern_return_t IOServiceGetMatchingServices(  mach_port_t mainPort,  CFDictionaryRef matching,  io_iterator_t * existing );
kern_return_t IOServiceAddNotification(  mach_port_t mainPort,  const io_name_t notificationType,  CFDictionaryRef matching,  mach_port_t wakePort,  uintptr_t reference,  io_iterator_t * notification ) __attribute__((deprecated));
kern_return_t IOServiceAddMatchingNotification(  IONotificationPortRef notifyPort,  const io_name_t notificationType,  CFDictionaryRef matching,         IOServiceMatchingCallback callback,         void * refCon,  io_iterator_t * notification );
CFMutableDictionaryRef IOServiceMatching(  const char * name );
CFMutableDictionaryRef IOServiceNameMatching(  const char * name );
CFMutableDictionaryRef IOBSDNameMatching(  mach_port_t mainPort,  uint32_t options,  const char * bsdName );
CFMutableDictionaryRef IOOpenFirmwarePathMatching(  mach_port_t mainPort,  uint32_t options,  const char * path );
CFMutableDictionaryRef IORegistryEntryIDMatching(  uint64_t entryID );
typedef struct __DASession * DASessionRef;
extern DASessionRef DASessionCreate( CFAllocatorRef allocator );
typedef struct __DADisk * DADiskRef;
extern DADiskRef DADiskCreateFromBSDName( CFAllocatorRef allocator, DASessionRef session, const char * name );
extern DADiskRef DADiskCreateFromIOMedia( CFAllocatorRef allocator, DASessionRef session, io_service_t media );
extern CFDictionaryRef DADiskCopyDescription( DADiskRef disk );
extern DADiskRef DADiskCopyWholeDisk( DADiskRef disk );
@interface NSTask : NSObject - (id)init;
@end                    typedef struct CGColorSpace *CGColorSpaceRef;
typedef struct CGImage *CGImageRef;
typedef struct CGLayer *CGLayerRef;
@interface NSResponder : NSObject <NSCoding> {
}
@end    @protocol NSAnimatablePropertyContainer      - (id)animator;
@end  extern NSString *NSAnimationTriggerOrderIn ;
@interface NSView : NSResponder  <NSAnimatablePropertyContainer>  {
}
@end @protocol NSValidatedUserInterfaceItem - (SEL)action;
@end   @protocol NSUserInterfaceValidations - (BOOL)validateUserInterfaceItem:(id <NSValidatedUserInterfaceItem>)anItem;
@end  @class NSDate, NSDictionary, NSError, NSException, NSNotification;
@interface NSApplication : NSResponder <NSUserInterfaceValidations> {
}
@end   enum {
NSTerminateCancel = 0,         NSTerminateNow = 1,         NSTerminateLater = 2 };
typedef NSUInteger NSApplicationTerminateReply;
@protocol NSApplicationDelegate <NSObject> @optional        - (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender;
@end  @class NSAttributedString, NSEvent, NSFont, NSFormatter, NSImage, NSMenu, NSText, NSView, NSTextView;
@interface NSCell : NSObject <NSCopying, NSCoding> {
}
@end @class NSTextField, NSPanel, NSArray, NSWindow, NSImage, NSButton, NSError;
typedef struct {
}
CVTimeStamp;
@interface CIImage : NSObject <NSCoding, NSCopying> {
}
typedef int CIFormat;
@end  enum {
kDAReturnSuccess = 0,     kDAReturnError = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x01,     kDAReturnBusy = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x02,     kDAReturnBadArgument = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x03,     kDAReturnExclusiveAccess = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x04,     kDAReturnNoResources = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x05,     kDAReturnNotFound = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x06,     kDAReturnNotMounted = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x07,     kDAReturnNotPermitted = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x08,     kDAReturnNotPrivileged = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x09,     kDAReturnNotReady = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x0A,     kDAReturnNotWritable = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x0B,     kDAReturnUnsupported = (((0x3eU)&0x3f)<<26) | (((0x368)&0xfff)<<14) | 0x0C };
typedef mach_error_t DAReturn;
typedef const struct __DADissenter * DADissenterRef;
extern DADissenterRef DADissenterCreate( CFAllocatorRef allocator, DAReturn status, CFStringRef string );
@interface CIContext: NSObject {
}
- (CGImageRef)createCGImage:(CIImage *)im fromRect:(CGRect)r;
- (CGImageRef)createCGImage:(CIImage *)im fromRect:(CGRect)r     format:(CIFormat)f colorSpace:(CGColorSpaceRef)cs;
- (CGLayerRef)createCGLayerWithSize:(CGSize)size info:(CFDictionaryRef)d;
@end extern NSString* const QCRendererEventKey;
@protocol QCCompositionRenderer - (NSDictionary*) attributes;
@end   @interface QCRenderer : NSObject <QCCompositionRenderer> {
}
- (id) createSnapshotImageOfType:(NSString*)type;
@end  extern NSString* const QCViewDidStartRenderingNotification;
@interface QCView : NSView <QCCompositionRenderer> {
}
- (id) createSnapshotImageOfType:(NSString*)type;
@end    enum {
ICEXIFOrientation1 = 1,     ICEXIFOrientation2 = 2,     ICEXIFOrientation3 = 3,     ICEXIFOrientation4 = 4,     ICEXIFOrientation5 = 5,     ICEXIFOrientation6 = 6,     ICEXIFOrientation7 = 7,     ICEXIFOrientation8 = 8, };
@class ICDevice;
@protocol ICDeviceDelegate <NSObject>  @required      - (void)didRemoveDevice:(ICDevice*)device;
@end extern NSString *const ICScannerStatusWarmingUp;
@class ICScannerDevice;
@protocol ICScannerDeviceDelegate <ICDeviceDelegate>  @optional       - (void)scannerDeviceDidBecomeAvailable:(ICScannerDevice*)scanner;
@end

typedef long unsigned int __darwin_size_t;
typedef __darwin_size_t size_t;
typedef unsigned long CFTypeID;
struct CGPoint {
  CGFloat x;
  CGFloat y;
};
typedef struct CGPoint CGPoint;
typedef struct CGGradient *CGGradientRef;
typedef uint32_t CGGradientDrawingOptions;
extern CFTypeID CGGradientGetTypeID(void);
extern CGGradientRef CGGradientCreateWithColorComponents(CGColorSpaceRef
  space, const CGFloat components[], const CGFloat locations[], size_t count);
extern CGGradientRef CGGradientCreateWithColors(CGColorSpaceRef space,
  CFArrayRef colors, const CGFloat locations[]);
extern CGGradientRef CGGradientRetain(CGGradientRef gradient);
extern void CGGradientRelease(CGGradientRef gradient);
typedef struct CGContext *CGContextRef;
extern void CGContextDrawLinearGradient(CGContextRef context,
    CGGradientRef gradient, CGPoint startPoint, CGPoint endPoint,
    CGGradientDrawingOptions options);
extern CGColorSpaceRef CGColorSpaceCreateDeviceRGB(void);

@interface NSMutableArray : NSObject
- (void)addObject:(id)object;
+ (id)array;
@end

enum {
    NSASCIIStringEncoding = 1,
    NSNEXTSTEPStringEncoding = 2,
    NSJapaneseEUCStringEncoding = 3,
    NSUTF8StringEncoding = 4,
    NSISOLatin1StringEncoding = 5,
    NSSymbolStringEncoding = 6,
    NSNonLossyASCIIStringEncoding = 7,
};
typedef struct __CFString * CFMutableStringRef;
typedef NSUInteger NSStringEncoding;

extern CFStringRef CFStringCreateWithCStringNoCopy(CFAllocatorRef alloc, const char *cStr, CFStringEncoding encoding, CFAllocatorRef contentsDeallocator);

typedef struct {
  int ref;
} isl_basic_map;

//===----------------------------------------------------------------------===//
// Test cases.
//===----------------------------------------------------------------------===//

void foo(id x) {
  [x retain];
}

void bar(id x) {
  [x release];
}

void test() {
  NSString *s = [[NSString alloc] init]; // expected-warning {{Potential leak}}
  foo(s);
  foo(s);
  bar(s);
}

void test_neg() {
  NSString *s = [[NSString alloc] init]; // no-warning  
  foo(s);
  foo(s);
  bar(s);
  bar(s);
  bar(s);
}

__attribute__((annotate("rc_ownership_returns_retained"))) isl_basic_map *isl_basic_map_cow(__attribute__((annotate("rc_ownership_consumed"))) isl_basic_map *bmap);
void free(void *);

void callee_side_parameter_checking_leak(__attribute__((annotate("rc_ownership_consumed"))) isl_basic_map *bmap) { // expected-warning {{Potential leak of an object}}
}

// As 'isl_basic_map_free' is annotated with 'rc_ownership_trusted_implementation', RetainCountChecker trusts its
// implementation and doesn't analyze its body. If the annotation 'rc_ownership_trusted_implementation' is removed,
// a leak warning is raised by RetainCountChecker as the analyzer is unable to detect a decrement in the reference
// count of 'bmap' along the path in 'isl_basic_map_free' assuming the predicate of the second 'if' branch to be
// true or assuming both the predicates in the function to be false.
__attribute__((annotate("rc_ownership_trusted_implementation"))) isl_basic_map *isl_basic_map_free(__attribute__((annotate("rc_ownership_consumed"))) isl_basic_map *bmap) {
  if (!bmap)
    return NULL;

  if (--bmap->ref > 0)
    return NULL;

  free(bmap);
  return NULL;
}

// As 'isl_basic_map_copy' is annotated with 'rc_ownership_trusted_implementation', RetainCountChecker trusts its
// implementation and doesn't analyze its body. If that annotation is removed, a 'use-after-release' warning might
// be raised by RetainCountChecker as the pointer which is passed as an argument to this function and the pointer
// which is returned from the function point to the same memory location.
__attribute__((annotate("rc_ownership_trusted_implementation"))) __attribute__((annotate("rc_ownership_returns_retained"))) isl_basic_map *isl_basic_map_copy(isl_basic_map *bmap) {
  if (!bmap)
    return NULL;

  bmap->ref++;
  return bmap;
}

void test_use_after_release_with_trusted_implementation_annotate_attribute(__attribute__((annotate("rc_ownership_consumed"))) isl_basic_map *bmap) {
  // After this call, 'bmap' has a +1 reference count.
  bmap = isl_basic_map_cow(bmap);
  // After the call to 'isl_basic_map_copy', 'bmap' has a +1 reference count.
  isl_basic_map *temp = isl_basic_map_cow(isl_basic_map_copy(bmap));
  // After this call, 'bmap' has a +0 reference count.
  isl_basic_map *temp2 = isl_basic_map_cow(bmap); // no-warning
  isl_basic_map_free(temp2);
  isl_basic_map_free(temp);
}

void test_leak_with_trusted_implementation_annotate_attribute(__attribute__((annotate("rc_ownership_consumed"))) isl_basic_map *bmap) {
  // After this call, 'bmap' has a +1 reference count.
  bmap = isl_basic_map_cow(bmap); // no-warning
  // After this call, 'bmap' has a +0 reference count.
  isl_basic_map_free(bmap);
}

void callee_side_parameter_checking_incorrect_rc_decrement(isl_basic_map *bmap) {
  isl_basic_map_free(bmap); // expected-warning {{Incorrect decrement of the reference count}}
}

__attribute__((annotate("rc_ownership_returns_retained"))) isl_basic_map *callee_side_parameter_checking_return_notowned_object(isl_basic_map *bmap) {
  return bmap; // expected-warning {{Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected}}
}

__attribute__((annotate("rc_ownership_returns_retained"))) isl_basic_map *callee_side_parameter_checking_assign_consumed_parameter_leak_return(__attribute__((annotate("rc_ownership_consumed"))) isl_basic_map *bmap1, __attribute__((annotate("rc_ownership_consumed"))) isl_basic_map *bmap2) { // expected-warning {{Potential leak of an object}}
  bmap1 = bmap2;
  isl_basic_map_free(bmap2);
  return bmap1;
}

__attribute__((annotate("rc_ownership_returns_retained"))) isl_basic_map *callee_side_parameter_checking_assign_consumed_parameter_leak(__attribute__((annotate("rc_ownership_consumed"))) isl_basic_map *bmap1, __attribute__((annotate("rc_ownership_consumed"))) isl_basic_map *bmap2) { // expected-warning {{Potential leak of an object}}
  bmap1 = bmap2;
  isl_basic_map_free(bmap1);
  return bmap2;
}

__attribute__((annotate("rc_ownership_returns_retained"))) isl_basic_map *error_path_leak(__attribute__((annotate("rc_ownership_consumed"))) isl_basic_map *bmap1, __attribute__((annotate("rc_ownership_consumed"))) isl_basic_map *bmap2) { // expected-warning {{Potential leak of an object}}
  bmap1 = isl_basic_map_cow(bmap1);
  if (!bmap1 || !bmap2)
    goto error;

  isl_basic_map_free(bmap2);
  return bmap1;
error:
  return isl_basic_map_free(bmap1);
}

//===----------------------------------------------------------------------===//
// Test returning retained and not-retained values.
//===----------------------------------------------------------------------===//

// On return (intraprocedural), assume CF objects are leaked.
CFStringRef test_return_ratained_CF(char *bytes) {
  CFStringRef str;
  return CFStringCreateWithCStringNoCopy(0, bytes, NSNEXTSTEPStringEncoding, 0); // expected-warning {{leak}}
}

// On return (intraprocedural), assume NSObjects are not leaked.
id test_return_retained_NS() {
  return [[NSString alloc] init]; // no-warning
}

void test_test_return_retained() {
  id x = test_return_retained_NS(); // expected-warning {{leak}}
  [x retain];
  [x release];
}

//===----------------------------------------------------------------------===//
// Test not applying "double effects" from inlining and RetainCountChecker summaries.
// If we inline a call, we should already see its retain/release semantics.
//===----------------------------------------------------------------------===//

__attribute__((cf_returns_retained)) CFStringRef test_return_inline(CFStringRef x) {
  CFRetain(x);
  return x;
}

void test_test_return_inline(char *bytes) {
  CFStringRef str = CFStringCreateWithCStringNoCopy(0, bytes, NSNEXTSTEPStringEncoding, 0);
  // After this call, 'str' really has +2 reference count.
  CFStringRef str2 = test_return_inline(str);
  // After this call, 'str' really has a +1 reference count.
  CFRelease(str);
  // After this call, 'str2' and 'str' has a +0 reference count.
  CFRelease(str2);
}

void test_test_return_inline_2(char *bytes) {
  CFStringRef str = CFStringCreateWithCStringNoCopy(0, bytes, NSNEXTSTEPStringEncoding, 0); // expected-warning {{leak}}
  // After this call, 'str' really has +2 reference count.
  CFStringRef str2 = test_return_inline(str);
  // After this call, 'str' really has a +1 reference count.
  CFRelease(str);
}

extern CFStringRef getString(void);
CFStringRef testCovariantReturnType(void) __attribute__((cf_returns_retained));

void usetestCovariantReturnType() {
  CFStringRef S = ((void*)0);
  S = testCovariantReturnType();
  if (S)
    CFRelease(S);
} 

CFStringRef testCovariantReturnType() {
  CFStringRef Str = ((void*)0);
  Str = getString();
  if (Str) {
    CFRetain(Str);
  }
  return Str;
}

// Test that we reanalyze ObjC methods which have been inlined. When reanalyzing
// them, make sure we inline very small functions.
id returnInputParam(id x) {
  return x;
}

@interface MyClass : NSObject
- (id)test_reanalyze_as_top_level;
- (void)test_inline_tiny_when_reanalyzing;
- (void)inline_test_reanalyze_as_top_level;
@end

@implementation MyClass
- (void)test_inline_tiny_when_reanalyzing {
  id x = [[NSString alloc] init]; // no-warning
  x = returnInputParam(x);
  [x release];
}

- (id)test_reanalyze_as_top_level {
  // This method does not follow naming conventions, so a warning will be
  // reported when it is reanalyzed at top level.
  return [[NSString alloc] init]; // expected-warning {{leak}}
}

- (void)inline_test_reanalyze_as_top_level {
  id x = [self test_reanalyze_as_top_level];
  [x release];
  [self test_inline_tiny_when_reanalyzing];
}
@end

// Original problem: rdar://problem/50739539
@interface MyClassThatLeaksDuringInit : NSObject

+ (MyClassThatLeaksDuringInit *)getAnInstance1;
+ (MyClassThatLeaksDuringInit *)getAnInstance2;

@end

@implementation MyClassThatLeaksDuringInit

+ (MyClassThatLeaksDuringInit *)getAnInstance1 {
  return [[[MyClassThatLeaksDuringInit alloc] init] autorelease]; // expected-warning{{leak}}
}

+ (MyClassThatLeaksDuringInit *)getAnInstance2 {
  return [[[[self class] alloc] init] autorelease]; // expected-warning{{leak}}
}

- (instancetype)init {
  if (1) {
    return nil;
  }

  if (nil != (self = [super init])) {
  }
  return self;
}

@end
