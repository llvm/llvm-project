// RUN: %clang_cc1 -arcmt-action=check -fobjc-arc -fobjc-runtime=macosx-10.8.0 -triple x86_64-apple-darwin12 -fblocks -Werror %s

#if __has_feature(objc_arc)
#define NS_AUTOMATED_REFCOUNT_UNAVAILABLE __attribute__((unavailable("not available in automatic reference counting mode")))
#else
#define NS_AUTOMATED_REFCOUNT_UNAVAILABLE
#endif

typedef const void * CFTypeRef;
CFTypeRef CFBridgingRetain(id X);
id CFBridgingRelease(CFTypeRef);

typedef int BOOL;
typedef unsigned NSUInteger;

@protocol NSObject
- (id)retain NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
- (NSUInteger)retainCount NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
- (oneway void)release NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
- (id)autorelease NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
@end

@interface NSObject <NSObject> {}
- (id)init;

+ (id)new;
+ (id)alloc;
- (void)dealloc;

- (void)finalize;

- (id)copy;
- (id)mutableCopy;
@end

typedef const struct __CFString * CFStringRef;
extern const CFStringRef kUTTypePlainText;
extern const CFStringRef kUTTypeRTF;
@class NSString;

@interface Test : NSObject
@property (weak) NSString *weakProperty;
@end

@implementation Test
@end

#if ! __has_feature(objc_arc)
#error This file must be compiled with ARC (set -fobjc_arc flag on file)
#endif
