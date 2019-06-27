@protocol Proto

@required
-(void)method:(int)x;

@end

@interface Base
@end

@interface I : Base<Proto>
@property int p1;
@property int p2;
@end

// Initiate the action within the @implementation
@implementation I

@dynamic p1;
@synthesize p2 = _p2;

- (void)anotherMethod {
  int x = 0;
}

void function(int x) {
  int y = x;
}

@end

// RUN: clang-refactor-test list-actions -at=%s:18:1 %s | FileCheck --check-prefix=CHECK-ACTION %s
// CHECK-ACTION: Add Missing Protocol Requirements

// Ensure the the action can be initiated in the @implementation / @interface:

// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -in=%s:11:1-27 -in=%s:12:1-18 -in=%s:13:1-18 -in=%s:14:1-5 %s | FileCheck --check-prefix=CHECKI1 %s
// CHECKI1: Initiated the 'fill-in-missing-protocol-stubs' action at [[@LINE-27]]:1
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -in=%s:17:1-18 -at=%s:18:1 -in=%s:19:1-13 -in=%s:20:1-22 -at=%s:21:1 -at=%s:25:1 -at=%s:29:1 -in=%s:30:1-5 %s | FileCheck --check-prefix=CHECK1 %s
// CHECK1: Initiated the 'fill-in-missing-protocol-stubs' action at [[@LINE-23]]:1

// Ensure that the action can't be initiated in other places:

// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -in=%s:1:1-10 -in=%s:4:1-21 -in=%s:6:1-5 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s
// CHECK-NO: Failed to initiate the refactoring action

// Ensure that the action can't be initiated in methods/functions in @implementation:

// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -in=%s:22:1-24 -in=%s:23:1-13 -in=%s:24:1-2 -in=%s:26:1-23 -in=%s:27:1-13 -in=%s:28:1-2 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

@protocol P2

-(void)method2:(int)y;

@end

@interface I (Category) <P2>

@end

@implementation I (Category)

- (void)anotherMethod2:(int)x {
  int x = 0;
}

void aFunction(int x) {
  int y = x;
}

@end

// Ensure the the action can be initiated in the category @implementation:

// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -in=%s:57:1-29 -at=%s:58:1 -in=%s:59:1-5 %s | FileCheck --check-prefix=CHECKI2 %s
// CHECKI2: Initiated the 'fill-in-missing-protocol-stubs' action at [[@LINE-19]]:1

// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -in=%s:61:1-29 -at=%s:62:1 -at=%s:66:1 -at=%s:70:1 -in=%s:71:1-5 %s | FileCheck --check-prefix=CHECK2 %s
// CHECK2: Initiated the 'fill-in-missing-protocol-stubs' action at [[@LINE-18]]:1

// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:60:1 -at=%s:72:1 -at=%s:73:1 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s
// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -in=%s:63:1-32 -in=%s:64:1-13 -in=%s:65:1-2 -in=%s:67:1-24 -in=%s:68:1-13 -in=%s:69:1-2 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s


// Check that initiation works with selection as well:

// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -selected=%s:17:1-30:5 -selected=%s:18:1-29:1 -selected=%s:20:1-20:10 -selected=%s:17:1-23:3 -selected=%s:27:3-30:5 -selected=%s:23:3-27:3 %s | FileCheck --check-prefix=CHECK1 %s

// Not when just one entire method is selected though!
// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -selected=%s:22:1-24:2 -selected=%s:26:1-28:2 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

// And not when the container is just partially selected!
// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -selected=%s:15:1-30:1 -selected=%s:17:1-40:1 -selected=%s:15:1-40:1 -selected=%s:1:1-90:1 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

@class ForwardClass;

// forward-class: +1:1
@implementation ForwardClass (ForwardClassCategory)
@end
// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=forward-class %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s
