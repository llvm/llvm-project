// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s

// Delta-Debugging reduced preamble.
typedef signed char BOOL;
typedef unsigned int NSUInteger;
@class NSString, Protocol;
extern void NSLog(NSString *format, ...) __attribute__((format(__NSString__, 1, 2)));
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@protocol NSCopying  - (id)copyWithZone:(NSZone *)zone; @end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder; @end
@interface NSObject <NSObject> {} 
+ (id)alloc; 
- (id)init;
@end
extern id NSAllocateObject(Class aClass, NSUInteger extraBytes, NSZone *zone);
@interface NSValue : NSObject <NSCopying, NSCoding>  - (void)getValue:(void *)value; @end
@class NSString, NSData;
typedef struct _NSPoint {} NSRange;
@interface NSValue (NSValueRangeExtensions) 
+ (NSValue *)valueWithRange:(NSRange)range;
- (id)objectAtIndex:(NSUInteger)index;
@end
@interface NSAutoreleasePool : NSObject {} - (void)drain; @end
extern NSString * const NSBundleDidLoadNotification;
typedef struct {} NSDecimal;
@interface NSNetService : NSObject {} - (id)init; @end
extern NSString * const NSUndoManagerCheckpointNotification;

int main (int argc, const char * argv[]) {
  NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
  id someUnintializedPointer = [someUnintializedPointer objectAtIndex:0]; // expected-warning{{Receiver in message expression is an uninitialized value}}
  NSLog(@"%@", someUnintializedPointer);    
  [pool drain];
  return 0;
}
