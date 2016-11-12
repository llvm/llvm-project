// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -analyze -analyzer-checker=core,osx.coreFoundation.CFRetainRelease,osx.cocoa.ClassRelease,osx.cocoa.RetainCount -fobjc-arc -fblocks -verify -Wno-objc-root-class %s -analyzer-output=text
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -analyze -analyzer-checker=core,osx.coreFoundation.CFRetainRelease,osx.cocoa.ClassRelease,osx.cocoa.RetainCount -fblocks -verify -Wno-objc-root-class %s -analyzer-output=text

#define HAS_ARC __has_feature(objc_arc)

typedef unsigned long long CFOptionFlags;
typedef signed long long CFIndex;

typedef CFIndex CFPropertyListFormat; enum {
    kCFPropertyListOpenStepFormat = 1,
    kCFPropertyListXMLFormat_v1_0 = 100,
    kCFPropertyListBinaryFormat_v1_0 = 200
};

typedef const struct __CFAllocator * CFAllocatorRef;
extern const CFAllocatorRef kCFAllocatorDefault;
typedef struct __CFDictionary * CFDictionaryRef;
typedef struct __CFError * CFErrorRef;
typedef struct __CFDataRef * CFDataRef;
typedef void * CFPropertyListRef;

CFPropertyListRef CFPropertyListCreateWithData(CFAllocatorRef allocator, CFDataRef data, CFOptionFlags options, CFPropertyListFormat *format, CFErrorRef *error);

typedef signed char BOOL;
typedef struct _NSZone NSZone;
@class NSDictionary;
@class NSData;
@class NSString;

@protocol NSObject
- (BOOL)isEqual:(id)object;
- (id)retain;
- (oneway void)release;
- (id)autorelease;
- (NSString *)description;
- (id)init;
@end
@interface NSObject <NSObject> {}
+ (id)allocWithZone:(NSZone *)zone;
+ (id)alloc;
+ (id)new;
- (void)dealloc;
@end

@interface NSDictionary : NSObject
@end

@interface SomeClass
@end

@implementation SomeClass
- (NSDictionary *)copyTestWithBridgeReturningRetainable:(NSData *)plistData {
  CFErrorRef error;
  CFDictionaryRef testDict = CFPropertyListCreateWithData(kCFAllocatorDefault, (__bridge CFDataRef)plistData, 0, 0, &error);
#if HAS_ARC
      // expected-note@-2 {{Call to function 'CFPropertyListCreateWithData' returns a Core Foundation object with a +1 retain count}}
#endif
  return (__bridge NSDictionary *)testDict;
#if HAS_ARC
      // expected-warning@-2 {{Potential leak of an object stored into 'testDict'}}
      // expected-note@-3 {{Object returned to caller as an owning reference (single retain count transferred to caller)}}
      // expected-note@-4 {{Object leaked: object allocated and stored into 'testDict' is returned from a method managed by Automatic Reference Counting}}
#endif
}

- (NSDictionary *)copyTestWithoutBridgeReturningRetainable:(NSData *)plistData {
  NSDictionary *testDict = [[NSDictionary alloc] init];
  return testDict; // no-warning
}

- (NSDictionary *)copyTestWithBridgeTransferReturningRetainable:(NSData *)plistData {
  CFErrorRef error;
  CFDictionaryRef testDict = CFPropertyListCreateWithData(kCFAllocatorDefault, (__bridge CFDataRef)plistData, 0, 0, &error);
  return (__bridge_transfer NSDictionary *)testDict; // no-warning under ARC
#if !HAS_ARC
      // expected-warning@-2 {{'__bridge_transfer' casts have no effect when not using ARC}} // Warning from Sema
#endif
}

- (CFDictionaryRef)copyTestReturningCoreFoundation:(NSData *)plistData {
  CFErrorRef error;
  CFDictionaryRef testDict = CFPropertyListCreateWithData(kCFAllocatorDefault, (__bridge CFDataRef)plistData, 0, 0, &error);
  return testDict;
}
@end

