// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx.cocoa.RetainCount,alpha.core -pedantic -verify -Wno-objc-root-class %s

// BEGIN delta-debugging reduced header stuff

typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSCoder;
@protocol NSObject
- (BOOL)isEqual:(id)object;
- (id)retain;
- (oneway void)release;
@end
@protocol NSCopying
- (id)copyWithZone:(NSZone *)zone;
@end
@protocol NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@interface NSObject <NSObject> {}
- (id)init;
+ (id)alloc;
@end
typedef double NSTimeInterval;
enum { NSAnimationEaseInOut, NSAnimationEaseIn, NSAnimationEaseOut, NSAnimationLinear };
typedef NSUInteger NSAnimationCurve;
@interface NSAnimation : NSObject <NSCopying, NSCoding> {}
- (id)initWithDuration:(NSTimeInterval)duration animationCurve:(NSAnimationCurve)animationCurve;
- (void)startAnimation;
- (void)setDelegate:(id)delegate;
@end

// END delta-debugging reduced header stuff

// From NSAnimation Class Reference
// -(void)startAnimation
// The receiver retains itself and is then autoreleased at the end 
// of the animation or when it receives stopAnimation.

@interface MyClass { }
- (void)animationDidEnd:(NSAnimation *)animation;
@end

@implementation MyClass
- (void)f1 {  
  // NOTE: The analyzer doesn't really handle this; it just stops tracking
  // 'animation' when it is sent the message 'setDelegate:'.
  NSAnimation *animation = [[NSAnimation alloc]   // no-warning
                            initWithDuration:1.0 
                            animationCurve:NSAnimationEaseInOut];
  
  [animation setDelegate:self];
  [animation startAnimation]; 
}

- (void)f2 {
  NSAnimation *animation = [[NSAnimation alloc]  // expected-warning{{leak}}
                            initWithDuration:1.0 
                            animationCurve:NSAnimationEaseInOut];

  [animation startAnimation]; 
}

- (void)animationDidEnd:(NSAnimation *)animation {
  [animation release];
}
@end
