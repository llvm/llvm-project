// RUN: %clang_cc1 -fsyntax-only -verify -fobjc-exceptions %s
// expected-no-diagnostics
# 1 "<command line>"
# 1 "/System/Library/Frameworks/Foundation.framework/Headers/Foundation.h" 1 3
typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;

@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;

@protocol NSObject
- (BOOL)isEqual:(id)object;
+ class;
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
@end

typedef struct {} NSFastEnumerationState;

@protocol NSFastEnumeration 
- (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id *)stackbuf count:(NSUInteger)len;
@end

@interface NSArray : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>
- (NSUInteger)count;
@end

@interface NSMutableArray : NSArray
- (void)addObject:(id)anObject;
+ (id)arrayWithCapacity:(int)numItems;
@end

@interface NSBundle : NSObject {}
+ (NSBundle *)bundleForClass:(Class)aClass;
- (NSString *)bundlePath;
- (void)setBundlePath:(NSString *)x;
@end

@interface NSException : NSObject <NSCopying, NSCoding> {}
@end

@class NSArray, NSDictionary, NSError, NSString, NSURL;

@interface DTPlugInManager : NSObject
@end

@implementation DTPlugInManager
+ (DTPlugInManager *)defaultPlugInManager {
  @try {
    NSMutableArray *plugInPaths = [NSMutableArray arrayWithCapacity:100];
    NSBundle *frameworkBundle = [NSBundle bundleForClass:[DTPlugInManager class]];
    frameworkBundle.bundlePath = 0;
    [plugInPaths addObject:frameworkBundle.bundlePath];
  }
  @catch (NSException *exception) {}
}
@end
