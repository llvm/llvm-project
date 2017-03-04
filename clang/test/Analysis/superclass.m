// RUN: %clang_analyze_cc1 -fblocks -analyzer-checker=osx.cocoa.MissingSuperCall -verify -Wno-objc-root-class %s

// Define used Classes
@protocol NSObject
- (id)retain;
- (oneway void)release;
@end
@interface NSObject <NSObject> {}
- (id)init;
+ (id)alloc;
@end
typedef char BOOL;
typedef double NSTimeInterval;
typedef enum UIViewAnimationOptions {
    UIViewAnimationOptionLayoutSubviews = 1 <<  0
} UIViewAnimationOptions;
@interface NSCoder : NSObject {}
@end

// Define the Superclasses for our Checks
@interface UIViewController : NSObject {}
- (void)addChildViewController:(UIViewController *)childController;
- (void)viewDidAppear:(BOOL)animated;
- (void)viewDidDisappear:(BOOL)animated;
- (void)viewDidUnload;
- (void)viewDidLoad;
- (void)viewWillUnload;
- (void)viewWillAppear:(BOOL)animated;
- (void)viewWillDisappear:(BOOL)animated;
- (void)didReceiveMemoryWarning;
- (void)removeFromParentViewController;
- (void)transitionFromViewController:(UIViewController *)fromViewController
  toViewController:(UIViewController *)toViewController
  duration:(NSTimeInterval)duration options:(UIViewAnimationOptions)options
  animations:(void (^)(void))animations
  completion:(void (^)(BOOL finished))completion;
@end
@interface UIResponder : NSObject {}
- (BOOL)resignFirstResponder;
@end
@interface NSResponder : NSObject {}
- (void)restoreStateWithCoder:(NSCoder *)coder;
- (void)encodeRestorableStateWithCoder:(NSCoder *)coder;
@end
@interface NSDocument : NSObject {}
- (void)restoreStateWithCoder:(NSCoder *)coder;
- (void)encodeRestorableStateWithCoder:(NSCoder *)coder;
@end

// Checks

// Do not warn if UIViewController/*Responder/NSDocument is not our superclass
@interface TestA 
@end
@implementation TestA

- (void)addChildViewController:(UIViewController *)childController {}
- (void)viewDidAppear:(BOOL)animated {}
- (void)viewDidDisappear:(BOOL)animated {}
- (void)viewDidUnload {}
- (void)viewDidLoad {}
- (void)viewWillUnload {}
- (void)viewWillAppear:(BOOL)animated {}
- (void)viewWillDisappear:(BOOL)animated {}
- (void)didReceiveMemoryWarning {}
- (void)removeFromParentViewController {}
- (BOOL)resignFirstResponder { return 0; }
- (void)restoreStateWithCoder:(NSCoder *)coder {}
- (void)encodeRestorableStateWithCoder:(NSCoder *)coder {}
@end

// Do not warn for the implementation in the superclass itself.
@implementation UIViewController
- (void)addChildViewController:(UIViewController *)childController {}
- (void)viewDidAppear:(BOOL)animated {}
- (void)viewDidDisappear:(BOOL)animated {}
- (void)viewDidUnload {}
- (void)viewDidLoad {}
- (void)viewWillUnload {}
- (void)viewWillAppear:(BOOL)animated {}
- (void)viewWillDisappear:(BOOL)animated {}
- (void)didReceiveMemoryWarning {}
- (void)removeFromParentViewController {}
- (void)transitionFromViewController:(UIViewController *)fromViewController
  toViewController:(UIViewController *)toViewController
  duration:(NSTimeInterval)duration options:(UIViewAnimationOptions)options
  animations:(void (^)(void))animations
  completion:(void (^)(BOOL finished))completion {}
@end

// Warn if UIViewController is our superclass and we do not call super
@interface TestB : UIViewController {}
@end
@implementation TestB

- (void)addChildViewController:(UIViewController *)childController {  
  int addChildViewController = 5;
  for (int i = 0; i < addChildViewController; i++)
  	[self viewDidAppear:i];
} // expected-warning {{The 'addChildViewController:' instance method in UIViewController subclass 'TestB' is missing a [super addChildViewController:] call}}
- (void)viewDidAppear:(BOOL)animated {} // expected-warning {{The 'viewDidAppear:' instance method in UIViewController subclass 'TestB' is missing a [super viewDidAppear:] call}}
- (void)viewDidDisappear:(BOOL)animated {} // expected-warning {{The 'viewDidDisappear:' instance method in UIViewController subclass 'TestB' is missing a [super viewDidDisappear:] call}}
- (void)viewDidUnload {} // expected-warning {{The 'viewDidUnload' instance method in UIViewController subclass 'TestB' is missing a [super viewDidUnload] call}}
- (void)viewDidLoad {} // expected-warning {{The 'viewDidLoad' instance method in UIViewController subclass 'TestB' is missing a [super viewDidLoad] call}}
- (void)viewWillUnload {} // expected-warning {{The 'viewWillUnload' instance method in UIViewController subclass 'TestB' is missing a [super viewWillUnload] call}}
- (void)viewWillAppear:(BOOL)animated {} // expected-warning {{The 'viewWillAppear:' instance method in UIViewController subclass 'TestB' is missing a [super viewWillAppear:] call}}
- (void)viewWillDisappear:(BOOL)animated {} // expected-warning {{The 'viewWillDisappear:' instance method in UIViewController subclass 'TestB' is missing a [super viewWillDisappear:] call}}
- (void)didReceiveMemoryWarning {} // expected-warning {{The 'didReceiveMemoryWarning' instance method in UIViewController subclass 'TestB' is missing a [super didReceiveMemoryWarning] call}}
- (void)removeFromParentViewController {} // expected-warning {{The 'removeFromParentViewController' instance method in UIViewController subclass 'TestB' is missing a [super removeFromParentViewController] call}}

// Do not warn for methods were it shouldn't
- (void)shouldAutorotate {}
@end

// Do not warn if UIViewController is our superclass but we did call super
@interface TestC : UIViewController {}
@end
@implementation TestC

- (BOOL)methodReturningStuff {
  return 1;
}

- (void)methodDoingStuff {
  [super removeFromParentViewController];
}

- (void)addChildViewController:(UIViewController *)childController {
  [super addChildViewController:childController];
}

- (void)viewDidAppear:(BOOL)animated {
  [super viewDidAppear:animated];
} 

- (void)viewDidDisappear:(BOOL)animated {
  [super viewDidDisappear:animated]; 
}

- (void)viewDidUnload {
  [super viewDidUnload];
}

- (void)viewDidLoad {
  [super viewDidLoad];
}

- (void)viewWillUnload {
  [super viewWillUnload];
} 

- (void)viewWillAppear:(BOOL)animated {
  int i = 0; // Also don't start warning just because we do additional stuff
  i++;
  [self viewDidDisappear:i];
  [super viewWillAppear:animated];
} 

- (void)viewWillDisappear:(BOOL)animated {
  [super viewWillDisappear:[self methodReturningStuff]];
}

- (void)didReceiveMemoryWarning {
  [super didReceiveMemoryWarning];
}

// We expect a warning here because at the moment the super-call can't be 
// done from another method.
- (void)removeFromParentViewController { 
  [self methodDoingStuff]; 
} // expected-warning {{The 'removeFromParentViewController' instance method in UIViewController subclass 'TestC' is missing a [super removeFromParentViewController] call}}
@end


// Do warn for UIResponder subclasses that don't call super
@interface TestD : UIResponder {}
@end
@implementation TestD

- (BOOL)resignFirstResponder {
  return 0;
} // expected-warning {{The 'resignFirstResponder' instance method in UIResponder subclass 'TestD' is missing a [super resignFirstResponder] call}}
@end

// Do not warn for UIResponder subclasses that do the right thing
@interface TestE : UIResponder {}
@end
@implementation TestE

- (BOOL)resignFirstResponder {
  return [super resignFirstResponder];
}
@end

// Do warn for NSResponder subclasses that don't call super
@interface TestF : NSResponder {}
@end
@implementation TestF

- (void)restoreStateWithCoder:(NSCoder *)coder {
} // expected-warning {{The 'restoreStateWithCoder:' instance method in NSResponder subclass 'TestF' is missing a [super restoreStateWithCoder:] call}}
- (void)encodeRestorableStateWithCoder:(NSCoder *)coder {
} // expected-warning {{The 'encodeRestorableStateWithCoder:' instance method in NSResponder subclass 'TestF' is missing a [super encodeRestorableStateWithCoder:] call}}
@end

// Do not warn for NSResponder subclasses that do the right thing
@interface TestG : NSResponder {}
@end
@implementation TestG

- (void)restoreStateWithCoder:(NSCoder *)coder {
	[super restoreStateWithCoder:coder];
}
- (void)encodeRestorableStateWithCoder:(NSCoder *)coder {
	[super encodeRestorableStateWithCoder:coder];
}
@end

// Do warn for NSDocument subclasses that don't call super
@interface TestH : NSDocument {}
@end
@implementation TestH

- (void)restoreStateWithCoder:(NSCoder *)coder {
} // expected-warning {{The 'restoreStateWithCoder:' instance method in NSDocument subclass 'TestH' is missing a [super restoreStateWithCoder:] call}}
- (void)encodeRestorableStateWithCoder:(NSCoder *)coder {
} // expected-warning {{The 'encodeRestorableStateWithCoder:' instance method in NSDocument subclass 'TestH' is missing a [super encodeRestorableStateWithCoder:] call}}
@end

// Do not warn for NSDocument subclasses that do the right thing
@interface TestI : NSDocument {}
@end
@implementation TestI

- (void)restoreStateWithCoder:(NSCoder *)coder {
	[super restoreStateWithCoder:coder];
}
- (void)encodeRestorableStateWithCoder:(NSCoder *)coder {
	[super encodeRestorableStateWithCoder:coder];
}
@end