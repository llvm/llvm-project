// RUN: %clang_cc1 -x objective-c -emit-llvm -debug-info-kind=limited < %s | FileCheck %s
// Test to check that "self" argument is assigned a location.
// CHECK: call void @llvm.dbg.declare(metadata ptr %self.addr, metadata [[SELF:![0-9]*]], metadata !{{.*}})
// CHECK: [[SELF]] = !DILocalVariable(name: "self", arg: 1,

@interface Foo 
-(void) Bar: (int)x ;
@end


@implementation Foo
-(void) Bar: (int)x 
{
}
@end

