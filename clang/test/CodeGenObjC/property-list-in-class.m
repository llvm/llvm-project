// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck %s
// CHECK: l_OBJC_$_PROP_LIST_C2" = internal global { i32, i32, [3 x %struct._prop_t] } { i32 16, i32 3

@protocol P 
@property int i;
@end

@protocol P1 
@property int i1;
@end

@protocol P2 < P1> 
@property int i2;
@end

@interface C1 { id isa; } @end

@interface C2 : C1 <P, P2> {
    int i;
}
@property int i2;
@end

@implementation C1
+(void)initialize { }
@end

@implementation C2
@synthesize i;
@synthesize i1;
@synthesize i2;
@end
