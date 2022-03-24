@interface NSSomeClass
  -(instancetype)init;
@end

// Extension, inspired by UIKit UIViewController.h
@interface NSSomeClass (UIContainerViewControllerCallbacks)

- (void)didMoveToParentViewController:(NSSomeClass *)parent;

@end

// Named "SomeClassRed" for ast node filtering in the test.
enum ColorEnum { SomeClassRed, SomeClassGreen, SomeClassBlue };