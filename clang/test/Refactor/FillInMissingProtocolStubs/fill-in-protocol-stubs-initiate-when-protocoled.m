@protocol Proto

@required
-(void)method:(int)x;

@required
- (void)method2:(int)y;

@end

@interface Base
@end

// Initiate when @implementation's interface has a suitable protocol.
@interface I1 : Base<Proto>
@end
// CHECK1: Initiated the 'fill-in-missing-protocol-stubs' action at [[@LINE-2]]:1
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:15:1 %s | FileCheck --check-prefix=CHECK1 %s

@implementation I1

@end
// CHECK2: Initiated the 'fill-in-missing-protocol-stubs' action at [[@LINE-3]]:1
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:20:1 %s | FileCheck --check-prefix=CHECK2 %s

@interface I2 : I1

@end
// CHECK3: Initiated the 'fill-in-missing-protocol-stubs' action at [[@LINE-3]]:1
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:26:1 %s | FileCheck --check-prefix=CHECK3 %s

@implementation I2
@end
// CHECK4: Initiated the 'fill-in-missing-protocol-stubs' action at [[@LINE-2]]:1
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:32:1 %s | FileCheck --check-prefix=CHECK4 %s

// Shouldn't initiate when the @interface is a forward declaration.
@class ForwardDecl;
// CHECK-FORWARD: Failed to initiate the refactoring action!
// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -in=%s:38:1-19 %s 2>&1 | FileCheck --check-prefix=CHECK-FORWARD %s

// Shouldn't initiate when the @interface has no protocols:

@interface I3 : Base
@end
@implementation I3
@end

@implementation I4
@end

// CHECK-CLASS-NO-PROTO: Failed to initiate the refactoring action!
// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -in=%s:11:1-16 -in=%s:12:1-5 -at=%s:44:1 -at=%s:46:1 -at=%s:49:1 %s 2>&1 | FileCheck --check-prefix=CHECK-CLASS-NO-PROTO %s

@protocol Proto2

@required
-(int)method3;

@end

// Initiate when the category has a suitable protocol:
@interface I3 (Category) <Proto2>
// CHECK5: Initiated the 'fill-in-missing-protocol-stubs' action at [[@LINE-1]]:1
@end
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:63:1 %s | FileCheck --check-prefix=CHECK5 %s

@implementation I3 (Category)
// CHECK6: Initiated the 'fill-in-missing-protocol-stubs' action at [[@LINE-1]]:1
@end
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:63:1 %s | FileCheck --check-prefix=CHECK5 %s

@interface I1 (Category) <Proto2>
// CHECK7: Initiated the 'fill-in-missing-protocol-stubs' action at [[@LINE-1]]:1
@end
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:63:1 %s | FileCheck --check-prefix=CHECK5 %s

@implementation I1 (Category)
// CHECK8: Initiated the 'fill-in-missing-protocol-stubs' action at [[@LINE-1]]:1
@end
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:63:1 %s | FileCheck --check-prefix=CHECK5 %s

// Shouldn't initiate when the category has no protocols (even when the class has them):
@interface I1 (Category2)
@end

@implementation I1 (Category2)
@end

@interface I3 (Category2)
@end

@implementation I3 (Category2)
@end

// CHECK-CAT-NO-PROTO: Failed to initiate the refactoring action!
// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:84:1 -at=%s:87:1 -at=%s:90:1 -at=%s:93:1 %s 2>&1 | FileCheck --check-prefix=CHECK-CAT-NO-PROTO %s
