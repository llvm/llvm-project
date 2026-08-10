// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core,osx.cocoa.VariadicMethodTypes -fblocks -verify %s

//===----------------------------------------------------------------------===//
// The following code is reduced using delta-debugging from
// Foundation.h (Mac OS X).
//
// It includes the basic definitions for the test cases below.
// Not directly including Foundation.h directly makes this test case 
// both svelte and portable to non-Mac platforms.
//===----------------------------------------------------------------------===//

#define nil (void*)0
typedef const struct __CFString * CFStringRef;
extern const CFStringRef kCGImageSourceShouldCache __attribute__((visibility("default")));
typedef signed char BOOL;
typedef struct _NSZone NSZone;
typedef unsigned int NSUInteger;
@protocol NSObject
- (BOOL)isEqual:(id)object;
- (oneway void)release;
- (id)retain;
- (id)autorelease;
@end
@protocol NSCopying
- (id)copyWithZone:(NSZone *)zone;
@end
@protocol NSMutableCopying
- (id)mutableCopyWithZone:(NSZone *)zone;
@end
@class NSCoder;
@protocol NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@interface NSObject <NSObject> {}
- (id)init;
+ (id)alloc;
@end
typedef struct {} NSFastEnumerationState;
@protocol NSFastEnumeration
- (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id *)stackbuf count:(NSUInteger)len;
@end
@interface NSArray : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>
@end
@interface NSArray (NSArrayCreation)
+ (id)arrayWithObjects:(id)firstObj, ... __attribute__((sentinel(0,1)));
- (id)initWithObjects:(id)firstObj, ... __attribute__((sentinel(0,1)));
@end
@interface NSDictionary : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>
@end
@interface NSDictionary (NSDictionaryCreation)
+ (id)dictionaryWithObjectsAndKeys:(id)firstObject, ... __attribute__((sentinel(0,1)));
- (id)initWithObjectsAndKeys:(id)firstObject, ... __attribute__((sentinel(0,1)));
@end
@interface NSSet : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>
@end
@interface NSSet (NSSetCreation)
+ (id)setWithObjects:(id)firstObj, ... __attribute__((sentinel(0,1)));
- (id)initWithObjects:(id)firstObj, ... __attribute__((sentinel(0,1)));
@end
@interface NSOrderedSet : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>
@end
@interface NSOrderedSet (NSOrderedSetCreation)
+ (id)orderedSetWithObjects:(id)firstObj, ... __attribute__((sentinel(0,1)));
- (id)initWithObjects:(id)firstObj, ... __attribute__((sentinel(0,1)));
@end
@protocol P;
@class C;

typedef struct FooType * __attribute__ ((NSObject)) FooType;
typedef struct BarType * BarType;


void f(id a, id<P> b, C* c, C<P> *d, FooType fooType, BarType barType) {
  [NSArray arrayWithObjects:@"Hello", a, b, c, d, nil];
  [NSArray arrayWithObjects:@"Foo", ^{}, nil];

  [NSArray arrayWithObjects:@"Foo", "Bar", "Baz", nil]; // expected-warning {{Argument to 'NSArray' method 'arrayWithObjects:' should be an Objective-C pointer type, not 'char *'}}
  [NSDictionary dictionaryWithObjectsAndKeys:@"Foo", "Bar", nil]; // expected-warning {{Argument to 'NSDictionary' method 'dictionaryWithObjectsAndKeys:' should be an Objective-C pointer type, not 'char *'}}
  [NSSet setWithObjects:@"Foo", "Bar", nil]; // expected-warning {{Argument to 'NSSet' method 'setWithObjects:' should be an Objective-C pointer type, not 'char *'}}
  [NSOrderedSet orderedSetWithObjects:@"Foo", "Bar", nil]; // expected-warning {{Argument to 'NSOrderedSet' method 'orderedSetWithObjects:' should be an Objective-C pointer type, not 'char *'}}

  [[[NSArray alloc] initWithObjects:@"Foo", "Bar", nil] autorelease]; // expected-warning {{Argument to 'NSArray' method 'initWithObjects:' should be an Objective-C pointer type, not 'char *'}}
  [[[NSDictionary alloc] initWithObjectsAndKeys:@"Foo", "Bar", nil] autorelease]; // expected-warning {{Argument to 'NSDictionary' method 'initWithObjectsAndKeys:' should be an Objective-C pointer type, not 'char *'}}
  [[[NSDictionary alloc] initWithObjectsAndKeys:@"Foo", (void*) 0, nil] autorelease]; // no-warning
  [[[NSDictionary alloc] initWithObjectsAndKeys:@"Foo", kCGImageSourceShouldCache, nil] autorelease]; // no-warning
  [[[NSDictionary alloc] initWithObjectsAndKeys:@"Foo", fooType, nil] autorelease]; // no-warning
  [[[NSDictionary alloc] initWithObjectsAndKeys:@"Foo", barType, nil] autorelease]; // expected-warning {{Argument to 'NSDictionary' method 'initWithObjectsAndKeys:' should be an Objective-C pointer type, not 'BarType'}}
  [[[NSSet alloc] initWithObjects:@"Foo", "Bar", nil] autorelease]; // expected-warning {{Argument to 'NSSet' method 'initWithObjects:' should be an Objective-C pointer type, not 'char *'}}
  [[[NSOrderedSet alloc] initWithObjects:@"Foo", "Bar", nil] autorelease]; // expected-warning {{Argument to 'NSOrderedSet' method 'initWithObjects:' should be an Objective-C pointer type, not 'char *'}}
}

// This previously crashed the variadic argument checker.
@protocol RDar9273215
- (void)rdar9273215:(id)x, ...;
@end

void test_rdar9273215(id<RDar9273215> y) {
  return [y rdar9273215:y, y];
}

