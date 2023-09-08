// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject
- (BOOL)isEqual:(id)object;
@end
@protocol NSCopying  - (id)copyWithZone:(NSZone *)zone;
@end
@protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone;
@end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@interface NSObject <NSObject> {}
@end
extern id NSAllocateObject(Class aClass, NSUInteger extraBytes, NSZone *zone);
typedef struct {
  id *itemsPtr;
  unsigned long *mutationsPtr;
} NSFastEnumerationState;
@protocol NSFastEnumeration
- (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id *)stackbuf count:(NSUInteger)len;
@end
@class NSString;
@interface NSArray : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>  - (NSUInteger)count;
@end
@interface NSMutableArray : NSArray  - (void)addObject:(id)anObject;
@end
extern NSString * const NSUndoManagerCheckpointNotification;
@interface NSValueTransformer : NSObject {} @end
@class FooModel;
@interface FooObject : NSObject <NSCopying> {}
@end
@interface FooNode : FooObject {}
- (NSArray *) children;
@end
typedef enum { Foo_HUH_NONE } FooHUHCode;
@interface FooPlaypenEntry : FooNode {
  NSMutableArray *_interestingChildren;
  FooHUHCode _HUH;
  __attribute__((objc_gc(weak))) FooPlaypenEntry *_mostInterestingChild;
  id _author;
}
@property(copy) NSString *author;
- (BOOL) isInteresting;
@end  NSString *FooHUHCodeToString(FooHUHCode HUH) { return 0; }
@interface FooHUHCodeToStringTransformer: NSValueTransformer {
}
@end  @implementation FooPlaypenEntry  @synthesize author = _author;
- (BOOL) isInteresting { return 1; }
- (NSArray *) interestingChildren {
  if (!_interestingChildren) {
    for (FooPlaypenEntry *child in [self children]) {
      if ([child isInteresting]) {
        if (!_mostInterestingChild)
          _mostInterestingChild = child;
        else if (child->_HUH > _mostInterestingChild->_HUH) 
          _mostInterestingChild = child;
      }
    }
  }
  return 0;
}
- (FooHUHCode) HUH {
  if (_HUH == Foo_HUH_NONE) {
    if (_mostInterestingChild)
      return [_mostInterestingChild HUH];
  }
  return 0;
}
@end

@interface Test1 {
@public
  id ivar __attribute__((objc_gc(weak)));
}
@property (assign) id prop __attribute((objc_gc(weak)));
@end
void test1(Test1 *t) {
  id *(__attribute__((objc_gc(strong))) x) = &t->ivar; // expected-warning {{initializing '__strong id *' with an expression of type '__weak id *' discards qualifiers}}
}
