// RUN: %clang_cc1 -analyze -analyzer-checker=core,osx.cocoa.RetainCount,alpha.core -verify %s -analyzer-constraints=range -analyzer-store=region
// expected-no-diagnostics

typedef struct objc_selector *SEL;
typedef signed char BOOL;
typedef int NSInteger;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@protocol NSCopying  - (id)copyWithZone:(NSZone *)zone; @end
@protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone; @end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder; @end
@interface NSObject <NSObject> {} - (id)init; @end
extern id NSAllocateObject(Class aClass, NSUInteger extraBytes, NSZone *zone);
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>
- (NSUInteger)length;
+ (id)stringWithUTF8String:(const char *)nullTerminatedCString;
@end extern NSString * const NSBundleDidLoadNotification;
@interface NSAssertionHandler : NSObject {}
+ (NSAssertionHandler *)currentHandler;
- (void)handleFailureInMethod:(SEL)selector object:(id)object file:(NSString *)fileName lineNumber:(NSInteger)line description:(NSString *)format,...;
- (void)handleFailureInFunction:(NSString *)functionName file:(NSString *)fileName lineNumber:(NSInteger)line description:(NSString *)format,...;
@end
extern NSString * const NSConnectionReplyMode;

//----------------------------------------------------------------------------//
// The following test case was filed in PR 2593:
//   http://llvm.org/bugs/show_bug.cgi?id=2593
//
// There should be no null dereference flagged by the checker because of
// NSParameterAssert and NSAssert.


@interface TestAssert : NSObject {}
@end

@implementation TestAssert

- (id)initWithPointer: (int*)x
{
  // Expansion of: NSParameterAssert( x != 0 );
  do { if (!((x != 0))) { [[NSAssertionHandler currentHandler] handleFailureInMethod:_cmd object:self file:[NSString stringWithUTF8String:"CFRetainRelease_NSAssertionHandler.m"] lineNumber:21 description:(@"Invalid parameter not satisfying: %s"), ("x != 0"), (0), (0), (0), (0)]; } } while(0);

  if( (self = [super init]) != 0 )
  {
    *x = 1; // no-warning
  }

  return self;
}

- (id)initWithPointer2: (int*)x
{
  // Expansion of: NSAssert( x != 0, @"" );
  do { if (!((x != 0))) { [[NSAssertionHandler currentHandler] handleFailureInMethod:_cmd object:self file:[NSString stringWithUTF8String:"CFRetainRelease_NSAssertionHandler.m"] lineNumber:33 description:((@"")), (0), (0), (0), (0), (0)]; } } while(0);  

  if( (self = [super init]) != 0 )
  {
    *x = 1; // no-warning
  }

  return self;
}

@end

void pointerFunction (int *x) {
  // Manual expansion of NSCAssert( x != 0, @"")
  do { if (!((x != 0))) { [[NSAssertionHandler currentHandler] handleFailureInFunction:[NSString stringWithUTF8String:__PRETTY_FUNCTION__] file:[NSString stringWithUTF8String:__FILE__] lineNumber:__LINE__ description:((@""))]; } } while(0);  

  *x = 1; // no-warning
}
