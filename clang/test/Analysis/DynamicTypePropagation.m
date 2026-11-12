// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx.cocoa.ObjCGenerics -verify %s

#if !__has_feature(objc_generics)
#  error Compiler does not support Objective-C generics?
#endif

typedef __typeof(sizeof(int)) size_t;
void *memset(void *, int, size_t);

#define nil 0
typedef unsigned long NSUInteger;
typedef int BOOL;

@protocol NSCopying
@end

__attribute__((objc_root_class))
@interface NSObject
- (void) myFunction:(int*)p myParam:(int) n;
@end

@interface MyType : NSObject <NSCopying>
- (void) myFunction:(int*)p myParam:(int) n;
@end

@interface NSArray<ObjectType> : NSObject
- (void) init;
- (BOOL)contains:(ObjectType)obj;
- (ObjectType)getObjAtIndex:(NSUInteger)idx;
- (ObjectType)objectAtIndexedSubscript:(NSUInteger)idx;
@property(readonly) ObjectType firstObject;
@end

@implementation NSObject
- (void) myFunction:(int*)p myParam:(int) n {
  (void)*p;// no warning
}
@end

@implementation MyType
- (void) myFunction:(int*)p myParam:(int) n {
  int i = 5/n;  // expected-warning {{}}
  (void)i;
}
@end

void testReturnType(NSArray<MyType *> *arr) {
  NSArray *erased = arr;
  NSObject *element = [erased firstObject];
  // TODO: myFunction currently dispatches to NSObject. Make it dispatch to
  // MyType instead!
  [element myFunction:0 myParam:0 ];
}

void testArgument(NSArray<MyType *> *arr, id element) {
  NSArray *erased = arr;
  [erased contains: element];
  // TODO: myFunction currently is not dispatched to MyType. Make it dispatch to
  // MyType!
  [element myFunction:0 myParam:0 ];
}

// Do not try this at home! The analyzer shouldn't crash though when it
// tries to figure out the dynamic type behind the alloca's return value.
void testAlloca(size_t NSArrayClassSizeWeKnowSomehow) {
  NSArray *arr = __builtin_alloca(NSArrayClassSizeWeKnowSomehow);
  memset(arr, 0, NSArrayClassSizeWeKnowSomehow);
  [arr init]; // no-crash
}
