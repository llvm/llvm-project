// RUN: rm -rf %t
// RUN: %clang_cc1 -objcmt-returns-innerpointer-property -objcmt-migrate-annotation -objcmt-migrate-readwrite-property -mt-migrate-directory %t %s -x objective-c -fobjc-runtime-has-weak -fobjc-arc -triple x86_64-apple-darwin11
// RUN: c-arcmt-test -mt-migrate-directory %t | arcmt-test -verify-transformed-files %s.result
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c -fobjc-runtime-has-weak -fobjc-arc %s.result

#ifndef NS_RETURNS_INNER_POINTER // defined in iOS 6 for sure
#define NS_RETURNS_INNER_POINTER __attribute__((objc_returns_inner_pointer))
#endif

#define CF_IMPLICIT_BRIDGING_ENABLED _Pragma("clang arc_cf_code_audited begin")

#define CF_IMPLICIT_BRIDGING_DISABLED _Pragma("clang arc_cf_code_audited end")

#if __has_feature(attribute_ns_returns_retained)
#define NS_RETURNS_RETAINED __attribute__((ns_returns_retained))
#endif
#if __has_feature(attribute_cf_returns_retained)
#define CF_RETURNS_RETAINED __attribute__((cf_returns_retained))
#endif
#if __has_feature(attribute_ns_returns_not_retained)
#define NS_RETURNS_NOT_RETAINED __attribute__((ns_returns_not_retained))
#endif
#if __has_feature(attribute_cf_returns_not_retained)
#define CF_RETURNS_NOT_RETAINED __attribute__((cf_returns_not_retained))
#endif
#if __has_feature(attribute_ns_consumes_self)
#define NS_CONSUMES_SELF __attribute__((ns_consumes_self))
#endif
#if __has_feature(attribute_ns_consumed)
#define NS_CONSUMED __attribute__((ns_consumed))
#endif
#if __has_feature(attribute_cf_consumed)
#define CF_CONSUMED __attribute__((cf_consumed))
#endif
#if __has_attribute(ns_returns_autoreleased)
#define NS_RETURNS_AUTORELEASED __attribute__((ns_returns_autoreleased))
#endif

#define NS_AVAILABLE __attribute__((availability(macosx,introduced=10.0)))

CF_IMPLICIT_BRIDGING_ENABLED

typedef unsigned long CFTypeID;
typedef unsigned long CFOptionFlags;
typedef unsigned long CFHashCode;

typedef signed long CFIndex; /*AnyObj*/
typedef const struct __CFArray * CFArrayRef;
typedef struct {
    CFIndex location;
    CFIndex length;
} CFRange;

typedef void (*CFArrayApplierFunction)(const void *value, void *context);

typedef enum CFComparisonResult : CFIndex CFComparisonResult; enum CFComparisonResult : CFIndex {
    kCFCompareLessThan = -1L,
    kCFCompareEqualTo = 0,
    kCFCompareGreaterThan = 1
};


typedef CFComparisonResult (*CFComparatorFunction)(const void *val1, const void *val2, void *context);

typedef struct __CFArray * CFMutableArrayRef;

typedef const struct __CFAttributedString *CFAttributedStringRef;
typedef struct __CFAttributedString *CFMutableAttributedStringRef;

typedef const struct __CFAllocator * CFAllocatorRef;

typedef const struct __CFString * CFStringRef;
typedef struct __CFString * CFMutableStringRef;

typedef const struct __CFDictionary * CFDictionaryRef;
typedef struct __CFDictionary * CFMutableDictionaryRef;

typedef struct CGImage *CGImageRef;

typedef struct OpaqueJSValue* JSObjectRef;

typedef JSObjectRef TTJSObjectRef;
typedef unsigned int NSUInteger;

CF_IMPLICIT_BRIDGING_DISABLED

@interface I
- (void*) ReturnsInnerPointer;
- (int*)  AlreadyReturnsInnerPointer NS_RETURNS_INNER_POINTER;
@end

@interface UIImage
- (CGImageRef)CGImage;
@end

@interface NSData
- (void *)bytes;
- (void **) ptr_bytes __attribute__((availability(macosx,unavailable)));
@end

@interface NSMutableData
- (void *)mutableBytes  __attribute__((deprecated)) __attribute__((unavailable));
@end

@interface JS
- (JSObjectRef)JSObject; 
- (TTJSObjectRef)JSObject1;
- (JSObjectRef*)JSObject2;
@end

// rdar://15044991
typedef void *SecTrustRef;

@interface NSURLProtectionSpace
@property (readonly) SecTrustRef serverTrust NS_AVAILABLE;
- (void *) FOO NS_AVAILABLE;
@property (readonly) void * mitTrust NS_AVAILABLE;

@property (readonly) void * mittiTrust;

@property (readonly) SecTrustRef XserverTrust;

- (SecTrustRef) FOO1 NS_AVAILABLE;

+ (const NSURLProtectionSpace *)ProtectionSpace;

// pointer personality functions
@property NSUInteger (*hashFunction)(const void *item, NSUInteger (*size)(const void *item));
@end
