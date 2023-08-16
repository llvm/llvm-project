@interface NSSomeClass
  -(instancetype)init;
@end

// Extension, inspired by UIKit UIViewController.h
@interface NSSomeClass (UIContainerViewControllerCallbacks)

- (void)didMoveToParentViewController:(NSSomeClass *)parent;

@end

// Named "SomeClassRed" for ast node filtering in the test.
enum ColorEnum { SomeClassRed, SomeClassGreen, SomeClassBlue };

#define CF_OPTIONS(_type, _name) _type __attribute__((availability(swift, unavailable))) _name; enum : _name
#define NS_OPTIONS(_type, _name) CF_OPTIONS(_type, _name)

typedef unsigned long NSUInteger;
typedef NS_OPTIONS(NSUInteger, NSSomeEnumOptions) {
	NSSomeEnumWithRed = 1,
	NSSomeEnumWithGreen,
	NSSomeEnumWithBlue,
};
