@protocol Proto

@required
-(void)method:(int)x;

@end

@protocol Proto2

@required
- (void)method2:(int)y;

@end

@interface Base
@end

// Initiate the action from extension if the @implementation is in the same TU.
@interface WithExtension: Base<Proto>
@end
@interface WithExtension()
@end
@interface WithExtension() <Proto2>
@end
@implementation WithExtension
// CHECK1: Initiated the 'fill-in-missing-protocol-stubs' action at [[@LINE-1]]:1
@end
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -in=%s:21:1-27 -in=%s:22:1-5 -in=%s:23:1-36 %s | FileCheck --check-prefix=CHECK1 %s

@interface WithoutImplementation: Base<Proto>
@end
@interface WithoutImplementation()
@end
@interface WithoutImplementation() <Proto2>
@end
// CHECK-NO-IMPL: Failed to initiate the refactoring action (Class extension without suitable @implementation)!
// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:32:1 -at=%s:34:1 %s 2>&1 | FileCheck --check-prefix=CHECK-NO-IMPL %s

// Initiate from the implementation even when the class has no protocols, but
// its extension does.

@interface NoProtocols: Base
@end
@interface NoProtocols() <Proto2>
@end
@implementation NoProtocols
@end
// CHECK2: Initiated the 'fill-in-missing-protocol-stubs' action at [[@LINE-2]]:1
// RUN: clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:44:1 -at=%s:46:1 %s | FileCheck --check-prefix=CHECK2 %s
// CHECK-NO-EXT-FROM-INTERFACE: Failed to initiate the refactoring action!
// RUN: not clang-refactor-test initiate -action fill-in-missing-protocol-stubs -at=%s:42:1 %s 2>&1 | FileCheck --check-prefix=CHECK-NO-EXT-FROM-INTERFACE %s
