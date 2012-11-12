// RUN: %clang_cc1 -analyze -analyzer-checker=core,osx.cocoa.RetainCount,osx.cocoa.NSAutoreleasePool -analyzer-store=region -fobjc-gc-only -fblocks -verify -Wno-objc-root-class %s

//===----------------------------------------------------------------------===//
// Header stuff.
//===----------------------------------------------------------------------===//

typedef unsigned int __darwin_natural_t;
typedef unsigned long uintptr_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef unsigned int UInt32;
typedef signed long CFIndex;
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
- (id)autorelease;
- (Class)class;
@end  @protocol NSCopying  - (id)copyWithZone:(NSZone *)zone;
@end  @protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone;
@end  @protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@interface NSObject <NSObject> {}
+ (id)allocWithZone:(NSZone *)zone;
+ (id)alloc;
- (void)dealloc;
- (oneway void)release;
- (id)copy;
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
@interface NSArray : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>
- (NSUInteger)count;
@end
@interface NSArray (NSArrayCreation)
+ (id)array;
+ (id)arrayWithObjects:(const id [])objects count:(NSUInteger)cnt;
@end
       @interface NSAutoreleasePool : NSObject {
}
- (void)drain;
- (id)init;
@end extern NSString * const NSBundleDidLoadNotification;
typedef double NSTimeInterval;
@interface NSDate : NSObject <NSCopying, NSCoding>  - (NSTimeInterval)timeIntervalSinceReferenceDate;
@end            typedef unsigned short unichar;
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>    - (NSUInteger)length;
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
io_service_t IOServiceGetMatchingService(  mach_port_t masterPort,  CFDictionaryRef matching );
kern_return_t IOServiceGetMatchingServices(  mach_port_t masterPort,  CFDictionaryRef matching,  io_iterator_t * existing );
kern_return_t IOServiceAddNotification(  mach_port_t masterPort,  const io_name_t notificationType,  CFDictionaryRef matching,  mach_port_t wakePort,  uintptr_t reference,  io_iterator_t * notification ) __attribute__((deprecated));
kern_return_t IOServiceAddMatchingNotification(  IONotificationPortRef notifyPort,  const io_name_t notificationType,  CFDictionaryRef matching,         IOServiceMatchingCallback callback,         void * refCon,  io_iterator_t * notification );
CFMutableDictionaryRef IOServiceMatching(  const char * name );
CFMutableDictionaryRef IOServiceNameMatching(  const char * name );
CFMutableDictionaryRef IOBSDNameMatching(  mach_port_t masterPort,  uint32_t options,  const char * bsdName );
CFMutableDictionaryRef IOOpenFirmwarePathMatching(  mach_port_t masterPort,  uint32_t options,  const char * path );
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
CFTypeRef CFMakeCollectable(CFTypeRef cf) ;

static __inline__ __attribute__((always_inline)) id NSMakeCollectable(CFTypeRef 
cf) {
    return cf ? (id)CFMakeCollectable(cf) : ((void*)0);
}

//===----------------------------------------------------------------------===//
// Test cases.
//===----------------------------------------------------------------------===//

void f1() {
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks); // no-warning
  id x = [(id) A autorelease];
  CFRelease((CFMutableArrayRef) x);
}

void f2() {
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks); // expected-warning{{leak}}
  id x = [(id) A retain];
  [x release];
  [x release];
}

void f3() {
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks); // expected-warning{{leak}}
  CFMakeCollectable(A);
  CFRetain(A);
}

void f3b() {
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks); // no-warning
  CFMakeCollectable(A);
}


void f4() {
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks); // expected-warning{{leak}}
  NSMakeCollectable(A);
  CFRetain(A);
}

void f4b() {
  CFMutableArrayRef A = CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks); // no-warning
  NSMakeCollectable(A);
}

void f5() {
  id x = [NSMakeCollectable(CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks)) autorelease]; // no-warning
}

void f5b() {
  id x = [(id) CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks) autorelease]; // expected-warning{{leak}}
}

// Test return of non-owned objects in contexts where an owned object
// is expected.
@interface TestReturnNotOwnedWhenExpectedOwned
- (NSString*)newString;
- (CFMutableArrayRef)newArray;
@end

@implementation TestReturnNotOwnedWhenExpectedOwned
- (NSString*)newString {
  NSString *s = [NSString stringWithUTF8String:"hello"]; // expected-warning{{Potential leak (when using garbage collection) of an object}}
  CFRetain(s);
  return s;
}
- (CFMutableArrayRef)newArray{
   return CFArrayCreateMutable(0, 10, &kCFTypeArrayCallBacks); // no-warning
}
@end

//===----------------------------------------------------------------------===//
// <rdar://problem/6948053> False positive: object substitution during -init*
//   methods warns about returning +0 when using -fobjc-gc-only
//===----------------------------------------------------------------------===//

@interface MyClassRdar6948053 : NSObject
- (id) init;
+ (id) shared;
@end

@implementation MyClassRdar6948053
+(id) shared {
  return (id) 0;
}
- (id) init
{
  Class myClass = [self class];  
  [self release];
  return [[myClass shared] retain]; // no-warning
}
@end

//===----------------------------------------------------------------------===//
// <rdar://problem/7174400> 'ciContext createCGImage:outputImage fromRect:' returns a retained CF object (not GC'ed)//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

void rdar_7174400(QCView *view, QCRenderer *renderer, CIContext *context,
                  NSString *str, CIImage *img, CGRect rect,
                  CIFormat form, CGColorSpaceRef cs) {
  [view createSnapshotImageOfType:str]; // no-warning
  [renderer createSnapshotImageOfType:str]; // no-warning
  [context createCGImage:img fromRect:rect]; // expected-warning{{leak}}
  [context createCGImage:img fromRect:rect format:form colorSpace:cs]; // expected-warning{{leak}}
}

//===----------------------------------------------------------------------===//
// <rdar://problem/6250216> Warn against using -[NSAutoreleasePool release] in 
//  GC mode
//===----------------------------------------------------------------------===//

void rdar_6250216(void) {
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
    [pool release]; // expected-warning{{Use -drain instead of -release when using NSAutoreleasePool and garbage collection}}
}


//===----------------------------------------------------------------------===//
// <rdar://problem/7407273> Don't crash when analyzing messages sent to blocks
//===----------------------------------------------------------------------===//

@class RDar7407273;
typedef void (^RDar7407273Block)(RDar7407273 *operation);
void rdar7407273(RDar7407273Block b) {
  [b copy];
}

//===----------------------------------------------------------------------===//
// Tests of ownership attributes.
//===----------------------------------------------------------------------===//

@interface TestOwnershipAttr : NSObject
- (NSString*) returnsAnOwnedString __attribute__((ns_returns_retained));
- (NSString*) returnsAnOwnedCFString  __attribute__((cf_returns_retained));
@end

void test_attr_1(TestOwnershipAttr *X) {
  NSString *str = [X returnsAnOwnedString]; // no-warning
}

void test_attr_1b(TestOwnershipAttr *X) {
  NSString *str = [X returnsAnOwnedCFString]; // expected-warning{{leak}}
}

@interface MyClassTestCFAttr : NSObject {}
- (NSDate*) returnsCFRetained __attribute__((cf_returns_retained));
- (NSDate*) alsoReturnsRetained;
- (NSDate*) returnsNSRetained __attribute__((ns_returns_retained));
@end

__attribute__((cf_returns_retained))
CFDateRef returnsRetainedCFDate()  {
  return CFDateCreate(0, CFAbsoluteTimeGetCurrent());
}

@implementation MyClassTestCFAttr
- (NSDate*) returnsCFRetained {
  return (NSDate*) returnsRetainedCFDate(); // No leak.
}

- (NSDate*) alsoReturnsRetained {
  return (NSDate*) returnsRetainedCFDate(); // expected-warning{{leak}}
}

- (NSDate*) returnsNSRetained {
  return (NSDate*) returnsRetainedCFDate(); // expected-warning{{leak}}
}
@end


#if __has_feature(attribute_ns_consumed)
#define NS_CONSUMED __attribute__((ns_consumed))
#endif
#if __has_feature(attribute_cf_consumed)
#define CF_CONSUMED __attribute__((cf_consumed))
#endif

void consumeAndStopTracking(id NS_CONSUMED obj, void (^callback)(void));
void CFConsumeAndStopTracking(CFTypeRef CF_CONSUMED obj, void (^callback)(void));

void testConsumeAndStopTracking() {
  id retained = [@[] retain]; // +0, GC
  consumeAndStopTracking(retained, ^{}); // no-warning

  id doubleRetained = [[@[] retain] retain]; // +0, GC
  consumeAndStopTracking(doubleRetained, ^{
    [doubleRetained release];
  }); // no-warning

  id unretained = @[]; // +0
  consumeAndStopTracking(unretained, ^{}); // no-warning, GC
}

void testCFConsumeAndStopTrackingMsg() {
  id retained = [@[] retain]; // +0, GC
  CFConsumeAndStopTracking((CFTypeRef)retained, ^{}); // expected-warning {{Incorrect decrement of the reference count of an object that is not owned at this point by the caller}}
}

void testCFConsumeAndStopTracking() {
  CFTypeRef retained = returnsRetainedCFDate(); // +1
  CFConsumeAndStopTracking(retained, ^{}); // no-warning

  CFTypeRef doubleRetained = CFRetain(returnsRetainedCFDate()); // +2
  CFConsumeAndStopTracking(doubleRetained, ^{
    CFRelease(doubleRetained);
  }); // no-warning

  id unretained = @[]; // +0
  CFConsumeAndStopTracking((CFTypeRef)unretained, ^{}); // expected-warning {{Incorrect decrement of the reference count of an object that is not owned at this point by the caller}}
}
