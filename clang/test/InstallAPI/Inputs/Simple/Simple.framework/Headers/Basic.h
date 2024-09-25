#import <Foundation/Foundation.h>

// Basic class with no super class
@interface Basic1
@end

@interface Basic2 : NSObject
@end

@interface Basic3 : NSObject
@property BOOL property1;
@property(readonly) BOOL property2;
@property(getter=isProperty3) BOOL property3;
@property BOOL dynamicProp;
@end

@interface Basic4 : NSObject {
@public
  BOOL ivar1;
@protected
  BOOL ivar2;
@package
  BOOL ivar3;
@private
  BOOL ivar4;
}
@end

__attribute__((visibility("hidden"))) @interface Basic4_1 : NSObject {
@public
  BOOL ivar1;
@protected
  BOOL ivar2;
@package
  BOOL ivar3;
@private
  BOOL ivar4;
}
@end

@interface Basic4_2 : NSObject {
@private
  BOOL ivar4;
@package
  BOOL ivar3;
@protected
  BOOL ivar2;
@public
  BOOL ivar1;
}
@end

@interface Basic5 : NSObject
+ (void)aClassMethod;
- (void)anInstanceMethod;
@end

@interface Basic6 : NSObject
@end

@interface Basic6 () {
@public
  BOOL ivar1;
}
@property BOOL property1;
- (void)anInstanceMethodFromAnExtension;
@end

@interface Basic6 (Foo)
@property BOOL property2;
- (void)anInstanceMethodFromACategory;
@end

__attribute__((visibility("hidden")))
@interface Basic7 : NSObject
@end

@interface Basic7 ()
- (void) anInstanceMethodFromAnHiddenExtension;
@end

@interface Basic8 : NSObject
+ (void)useSameName;
@end

// Classes and protocols can have the same name. For now they would only clash
// in the selector map if the protocl starts with '_'.
@protocol _A
- (void)aMethod;
@end

@interface A : NSObject
- (void)aMethod NS_AVAILABLE(10_11, 9_0);
- (void)bMethod NS_UNAVAILABLE;
@end

@interface Basic9 : NSObject
@property(readonly) BOOL aProperty NS_AVAILABLE(10_10, 8_0);
@end

@interface Basic9 (deprecated)
@property(readwrite) BOOL aProperty NS_DEPRECATED_MAC(10_8, 10_10);
@end
