// RUN: %clang_cc1 -fblocks -analyze -analyzer-checker=alpha.osx.cocoa.MissingSuperCall -verify -Wno-objc-root-class %s

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

// Do not warn if UIViewController isn't our superclass
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
- (void)shouldAutorotate {}; 
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
