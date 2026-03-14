// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify %s
// expected-no-diagnostics
// This program tests that if class implements the forwardInvocation method, then
// every method possible is implemented in the class and should not issue
// warning of the "Method definition not found" kind. */

@interface NSObject
@end

@interface NSInvocation
@end

@interface NSProxy
@end

@protocol MyProtocol
        -(void) doSomething;
@end

@interface DestinationClass : NSObject<MyProtocol>
        -(void) doSomething;
@end

@implementation DestinationClass
        -(void) doSomething
        {
        }
@end

@interface MyProxy : NSProxy<MyProtocol>
{
        DestinationClass        *mTarget;
}
        - (id) init;
        - (void)forwardInvocation:(NSInvocation *)anInvocation;
@end

@implementation MyProxy
        - (void)forwardInvocation:(NSInvocation *)anInvocation
        {
        }
	- (id) init { return 0; }
@end
