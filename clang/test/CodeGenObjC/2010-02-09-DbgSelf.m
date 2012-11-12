// RUN: %clang_cc1 -x objective-c -emit-llvm -g < %s | grep  "\"self\", metadata" 
// Test to check that "self" argument is assigned a location.

@interface Foo 
-(void) Bar: (int)x ;
@end


@implementation Foo
-(void) Bar: (int)x 
{
}
@end

