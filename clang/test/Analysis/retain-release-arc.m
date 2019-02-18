// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core,osx.coreFoundation.CFRetainRelease,osx.cocoa.ClassRelease,osx.cocoa.RetainCount -fobjc-arc -fblocks -verify -Wno-objc-root-class %s -analyzer-output=text
// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core,osx.coreFoundation.CFRetainRelease,osx.cocoa.ClassRelease,osx.cocoa.RetainCount -fblocks -verify -Wno-objc-root-class %s -analyzer-output=text

typedef __typeof(sizeof(int)) size_t;

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

#define OS_OBJECT_RETURNS_RETAINED __attribute__((__ns_returns_retained__))
#define DISPATCH_RETURNS_RETAINED OS_OBJECT_RETURNS_RETAINED

@protocol OS_dispatch_object
@end
@protocol OS_dispatch_data <OS_dispatch_object>
@end
@protocol OS_dispatch_queue <OS_dispatch_object>
@end

typedef NSObject<OS_dispatch_object> *dispatch_object_t;
typedef NSObject<OS_dispatch_data> *dispatch_data_t;
typedef NSObject<OS_dispatch_queue> *dispatch_queue_t;

typedef void (^dispatch_block_t)(void);

dispatch_queue_t dispatch_get_main_queue(void);

DISPATCH_RETURNS_RETAINED dispatch_data_t
dispatch_data_create(const void *buffer, size_t size,
                     dispatch_queue_t _Nullable queue,
                     dispatch_block_t _Nullable destructor);

void _dispatch_object_validate(dispatch_object_t object);

#define dispatch_retain(object) \
  __extension__({ dispatch_object_t _o = (object); \
                  _dispatch_object_validate(_o); \
                  (void)[_o retain]; })
#define dispatch_release(object) \
  __extension__({ dispatch_object_t _o = (object); \
                  _dispatch_object_validate(_o); \
                  [_o release]; })


@interface SomeClass
@end

@implementation SomeClass
- (NSDictionary *)copyTestWithBridgeReturningRetainable:(NSData *)plistData {
  CFErrorRef error;
  CFDictionaryRef testDict = CFPropertyListCreateWithData(kCFAllocatorDefault, (__bridge CFDataRef)plistData, 0, 0, &error);
#if HAS_ARC
      // expected-note@-2 {{Call to function 'CFPropertyListCreateWithData' returns a Core Foundation object of type 'CFPropertyListRef' with a +1 retain count}}
#endif
  return (__bridge NSDictionary *)testDict;
#if HAS_ARC
      // expected-warning@-2 {{Potential leak of an object stored into 'testDict'}}
      // expected-note@-3 {{Object leaked: object allocated and stored into 'testDict' is returned from a method managed by Automatic Reference Counting}}
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

int buf[1024];

void libdispatch_leaked_data() {
  dispatch_data_t data = dispatch_data_create(buf, 1024,
                                              dispatch_get_main_queue(), ^{});
}
#if !HAS_ARC
  // expected-warning@-2{{Potential leak of an object stored into 'data'}}
  // expected-note@-5{{Call to function 'dispatch_data_create' returns an Objective-C object with a +1 retain count}}
  // expected-note@-4{{Object leaked: object allocated and stored into 'data' is not referenced later in this execution path and has a retain count of +1}}
#endif

void libdispatch_dispatch_released_data() {
  dispatch_data_t data = dispatch_data_create(buf, 1024,
                                              dispatch_get_main_queue(), ^{});
#if !HAS_ARC
  dispatch_release(data); // no-warning
#endif
}

void libdispatch_objc_released_data() {
  dispatch_data_t data = dispatch_data_create(buf, 1024,
                                              dispatch_get_main_queue(), ^{});
#if !HAS_ARC
  [data release]; // no-warning
#endif
}

void libdispatch_leaked_retained_data() {
  dispatch_data_t data = dispatch_data_create(buf, 1024,
                                              dispatch_get_main_queue(), ^{});
#if !HAS_ARC
  dispatch_retain(data);
  [data release];
#endif
}
#if !HAS_ARC
// expected-warning@-2{{Potential leak of an object stored into 'data'}}
// expected-note@-9{{Call to function 'dispatch_data_create' returns an Objective-C object with a +1 retain count}}
// expected-note@-7{{Reference count incremented. The object now has a +2 retain count}}
// expected-note@-7{{Reference count decremented. The object now has a +1 retain count}}
// expected-note@-6{{Object leaked: object allocated and stored into 'data' is not referenced later in this execution path and has a retain count of +1}}
#endif
