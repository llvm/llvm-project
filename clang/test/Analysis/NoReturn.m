// RUN: %clang_cc1 -analyze -analyzer-checker=core,alpha.core -analyzer-store=region -analyzer-constraints=range -verify %s
// expected-no-diagnostics

#include <stdarg.h>

//===----------------------------------------------------------------------===//
// The following code is reduced using delta-debugging from
// Foundation.h (Mac OS X).
//
// It includes the basic definitions for the test cases below.
// Not directly including Foundation.h directly makes this test case 
// both svelte and portable to non-Mac platforms.
//===----------------------------------------------------------------------===//

typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object;
@end  @protocol NSCopying  - (id)copyWithZone:(NSZone *)zone;
@end  @protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone; @end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder; @end
@interface NSObject <NSObject> {} @end
extern id NSAllocateObject(Class aClass, NSUInteger extraBytes, NSZone *zone);
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>
- (NSUInteger)length;
+ (id)stringWithFormat:(NSString *)format, ...;
@end
@interface NSSimpleCString : NSString {} @end
@interface NSConstantString : NSSimpleCString @end
extern void *_NSConstantStringClassReference;
typedef double NSTimeInterval;
@interface NSDate : NSObject <NSCopying, NSCoding>  - (NSTimeInterval)timeIntervalSinceReferenceDate; @end
@class NSString, NSDictionary, NSArray;
@interface NSException : NSObject <NSCopying, NSCoding> {}
+ (NSException *)exceptionWithName:(NSString *)name reason:(NSString *)reason userInfo:(NSDictionary *)userInfo;
- (void)raise;
@end
@interface NSException (NSExceptionRaisingConveniences)
+ (void)raise:(NSString *)name format:(NSString *)format, ...;
+ (void)raise:(NSString *)name format:(NSString *)format arguments:(va_list)argList;
@end

enum {NSPointerFunctionsStrongMemory = (0 << 0),     NSPointerFunctionsZeroingWeakMemory = (1 << 0),     NSPointerFunctionsOpaqueMemory = (2 << 0),     NSPointerFunctionsMallocMemory = (3 << 0),     NSPointerFunctionsMachVirtualMemory = (4 << 0),        NSPointerFunctionsObjectPersonality = (0 << 8),     NSPointerFunctionsOpaquePersonality = (1 << 8),     NSPointerFunctionsObjectPointerPersonality = (2 << 8),     NSPointerFunctionsCStringPersonality = (3 << 8),     NSPointerFunctionsStructPersonality = (4 << 8),     NSPointerFunctionsIntegerPersonality = (5 << 8),      NSPointerFunctionsCopyIn = (1 << 16), };

//===----------------------------------------------------------------------===//
// Test cases.
//===----------------------------------------------------------------------===//

int f1(int *x, NSString* s) {
  
  if (x) ++x;
  
  [NSException raise:@"Blah" format:[NSString stringWithFormat:@"Blah %@", s]];
  
  return *x; // no-warning
}

int f2(int *x, ...) {
  
  if (x) ++x;
  va_list alist;
  va_start(alist, x);
  
  [NSException raise:@"Blah" format:@"Blah %@" arguments:alist];
  
  return *x; // no-warning
}

int f3(int* x) {
  
  if (x) ++x;
  
  [[NSException exceptionWithName:@"My Exception" reason:@"Want to test exceptions." userInfo:0] raise];

  return *x; // no-warning
}


@interface CustomException : NSException
@end

int testCustomException(int *x) {
  if (x != 0) return 0;

  [CustomException raise:@"Blah" format:@"abc"];

  return *x; // no-warning
}

