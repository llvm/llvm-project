// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx.cocoa.RetainCount -analyzer-store=region -Wno-objc-root-class -verify %s
// expected-no-diagnostics


//===----------------------------------------------------------------------===//
// The following code is reduced using delta-debugging from
// Foundation.h (Mac OS X).
//
// It includes the basic definitions for the test cases below.
// Not directly including Foundation.h directly makes this test case 
// both svelte and portable to non-Mac platforms.
//===----------------------------------------------------------------------===//

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
typedef struct objc_selector *SEL;
@class NSString, Protocol;
extern void NSLog(NSString *format, ...) __attribute__((format(__NSString__, 1, 2)));
typedef NSInteger NSComparisonResult;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject
- (BOOL)isEqual:(id)object;
- (oneway void)release;
- (Class)class;
- (id)retain;
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
+ (Class)class;
- (void)performSelectorOnMainThread:(SEL)aSelector withObject:(id)arg waitUntilDone:(BOOL)wait;
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
@end
@interface NSSimpleCString : NSString {} @end
@interface NSConstantString : NSSimpleCString @end
extern void *_NSConstantStringClassReference;

//===----------------------------------------------------------------------===//
// Test cases.
//===----------------------------------------------------------------------===//

//  <rdar://problem/6062730>
// The analyzer doesn't perform any inter-procedural analysis, so delegates
// involving [NSObject performSelector...] tend to lead to false positives.
// For now the analyzer just stops tracking the reference count of the
// receiver until we have better support for delegates.

@interface test_6062730 : NSObject
+ (void)postNotification:(NSString *)str;
- (void)foo;
- (void)bar;
@end

@implementation test_6062730
- (void) foo {
  NSString *str = [[NSString alloc] init]; // no-warning
  [test_6062730 performSelectorOnMainThread:@selector(postNotification:) withObject:str waitUntilDone:1];
}

- (void) bar {
  NSString *str = [[NSString alloc] init]; // no-warning
  [[self class] performSelectorOnMainThread:@selector(postNotification:) withObject:str waitUntilDone:1];
}

+ (void) postNotification:(NSString *)str {
  [str release]; // no-warning
}
@end


@interface ObjectThatRequiresDelegate : NSObject
- (id)initWithDelegate:(id)delegate;
- (id)initWithNumber:(int)num delegate:(id)delegate;
@end


@interface DelegateRequirerTest
@end
@implementation DelegateRequirerTest

- (void)test {
  (void)[[ObjectThatRequiresDelegate alloc] initWithDelegate:self];
  (void)[[ObjectThatRequiresDelegate alloc] initWithNumber:0 delegate:self];
  // no leak warnings -- these objects could be released in callback methods
}

@end
