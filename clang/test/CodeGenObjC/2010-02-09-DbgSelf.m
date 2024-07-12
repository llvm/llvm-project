// RUN: %clang_cc1 -x objective-c -emit-llvm -debug-info-kind=limited < %s | FileCheck %s
// Test to check that "self" argument is assigned a location.
// CHECK: #dbg_declare(ptr %self.addr, [[SELF:![0-9]*]], !{{.*}})
// CHECK: [[SELF]] = !DILocalVariable(name: "self", arg: 1,

@interface Foo 
-(void) Bar: (int)x ;
@end


@implementation Foo
-(void) Bar: (int)x 
{
}
@end

