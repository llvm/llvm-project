// RUN: %check_clang_tidy %s google-objc-avoid-nsobject-new %t

@interface NSObject
+ (instancetype)new;
+ (instancetype)alloc;
- (instancetype)init;
@end

@interface NSProxy  // Root class with no -init method.
@end

// NSDate provides a specific factory method.
@interface NSDate : NSObject
+ (instancetype)date;
@end

// For testing behavior with Objective-C Generics.
@interface NSMutableDictionary<__covariant KeyType, __covariant ObjectType> : NSObject
@end

@class NSString;

#define ALLOCATE_OBJECT(_Type) [_Type new]

void CheckSpecificInitRecommendations(void) {
  NSObject *object = [NSObject new];
  // CHECK-MESSAGES: [[@LINE-1]]:22: warning: do not create objects with +new [google-objc-avoid-nsobject-new]
  // CHECK-FIXES: NSObject *object = {{[[][[]}}NSObject alloc] init];

  NSDate *correctDate = [NSDate date];
  NSDate *incorrectDate = [NSDate new];
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: do not create objects with +new [google-objc-avoid-nsobject-new]
  // CHECK-FIXES: NSDate *incorrectDate = [NSDate date];

  NSObject *macroCreated = ALLOCATE_OBJECT(NSObject);  // Shouldn't warn on macros.

  NSMutableDictionary *dict = [NSMutableDictionary<NSString *, NSString *> new];
  // CHECK-MESSAGES: [[@LINE-1]]:31: warning: do not create objects with +new [google-objc-avoid-nsobject-new]
  // CHECK-FIXES: NSMutableDictionary *dict = {{[[][[]}}NSMutableDictionary<NSString *, NSString *> alloc] init];
}

@interface Foo : NSObject
+ (instancetype)new; // Declare again to suppress warning.
- (instancetype)initWithInt:(int)anInt;
- (instancetype)init __attribute__((unavailable));

- (id)new;
@end

@interface Baz : Foo // Check unavailable -init through inheritance.
@end

@interface ProxyFoo : NSProxy
+ (instancetype)new;
@end

void CallNewWhenInitUnavailable(void) {
  Foo *foo = [Foo new];
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: do not create objects with +new [google-objc-avoid-nsobject-new]

  Baz *baz = [Baz new];
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: do not create objects with +new [google-objc-avoid-nsobject-new]

  // Instance method -new calls may be weird, but are not strictly forbidden.
  Foo *bar = [[Foo alloc] initWithInt:4];
  [bar new];

  ProxyFoo *proxy = [ProxyFoo new];
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: do not create objects with +new [google-objc-avoid-nsobject-new]
}

@interface HasNewOverride : NSObject
@end

@implementation HasNewOverride
+ (instancetype)new {
  return [[self alloc] init];
}
// CHECK-MESSAGES: [[@LINE-3]]:1: warning: classes should not override +new [google-objc-avoid-nsobject-new]
@end
