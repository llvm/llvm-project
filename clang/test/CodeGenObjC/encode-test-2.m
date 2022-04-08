// RUN: %clang_cc1 -no-opaque-pointers -triple=i686-apple-darwin9 -emit-llvm -o - %s | FileCheck %s

// CHECK: private unnamed_addr constant [7 x i8] c"@\22<X>\22\00",
// CHECK: private unnamed_addr constant [10 x i8] c"@\22<X><Y>\22\00",
// CHECK: private unnamed_addr constant [13 x i8] c"@\22<X><Y><Z>\22\00",
// CHECK: private unnamed_addr constant [16 x i8] c"@\22Foo<X><Y><Z>\22\00",
// CHECK: private unnamed_addr constant [13 x i8] c"{Intf=@@@@#}\00",

// CHECK: @[[PROP_NAME_ATTR:.*]] = private unnamed_addr constant [5 x i8] c"T@,D\00",
// CHECK: @"_OBJC_$_PROP_LIST_C0" = internal global { i32, i32, [1 x %{{.*}}] } { i32 8, i32 1, [1 x %{{.*}}] [%{{.*}} { {{.*}}, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @[[PROP_NAME_ATTR]], i32 0, i32 0) }] },

@protocol X, Y, Z;
@class Foo;

@protocol Proto
@end

@interface Intf <Proto>
{
id <X> IVAR_x;
id <X, Y> IVAR_xy;
id <X, Y, Z> IVAR_xyz;
Foo <X, Y, Z> *IVAR_Fooxyz;
Class <X> IVAR_Classx;
}
@end

@implementation Intf 
@end

int main(void)
{
	const char * en = @encode(Intf);
}

@protocol P0
@property id prop0;
@end

@protocol P1 <P0>
@property id prop0;
@end

@interface C0 <P1>
@end

@implementation C0
@dynamic prop0;
@end
