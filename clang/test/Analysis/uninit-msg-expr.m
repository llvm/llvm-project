// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-store=region -verify %s

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
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@protocol NSCopying  - (id)copyWithZone:(NSZone *)zone; @end
@protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone; @end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder; @end
@interface NSObject <NSObject> {} @end
@class NSString, NSData;
@class NSString, NSData, NSMutableData, NSMutableDictionary, NSMutableArray;
typedef struct {} NSFastEnumerationState;
@protocol NSFastEnumeration
- (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id *)stackbuf count:(NSUInteger)len;
@end
@class NSData, NSIndexSet, NSString, NSURL;
@interface NSArray : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>
- (NSUInteger)count;
@end
@interface NSArray (NSArrayCreation)
+ (id)array;
- (NSUInteger)length;
- (void)addObject:(id)object;
@end
extern NSString * const NSUndoManagerCheckpointNotification;

//===----------------------------------------------------------------------===//
// Test cases.
//===----------------------------------------------------------------------===//

unsigned f1() {
  NSString *aString;
  return [aString length]; // expected-warning {{Receiver in message expression is an uninitialized value}}
}

unsigned f2() {
  NSString *aString = 0;
  return [aString length]; // no-warning
}

void f3() {
  NSMutableArray *aArray = [NSArray array];
  NSString *aString;
  [aArray addObject:aString]; // expected-warning {{1st argument in message expression is an uninitialized value}}
}
