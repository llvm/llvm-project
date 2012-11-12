// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics

@interface TestClass 
{
 int _isItIsOrIsItAint;
}
@property (readonly) int itIsOrItAint;
-(void) doSomething;
@end

@interface TestClass()
@property (readwrite) int itIsOrItAint;
@end

@implementation TestClass
@synthesize itIsOrItAint = _isItIsOrIsItAint;

-(void) doSomething
{
  int i = [self itIsOrItAint];

 [self setItIsOrItAint:(int)1];
}
@end
