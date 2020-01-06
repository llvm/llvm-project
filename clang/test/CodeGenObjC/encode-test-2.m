// RUN: %clang_cc1 -triple=i686-apple-darwin9 -emit-llvm -o - %s | FileCheck %s

// CHECK: private unnamed_addr constant [7 x i8] c"@\22<X>\22\00",
// CHECK: private unnamed_addr constant [10 x i8] c"@\22<X><Y>\22\00",
// CHECK: private unnamed_addr constant [13 x i8] c"@\22<X><Y><Z>\22\00",
// CHECK: private unnamed_addr constant [16 x i8] c"@\22Foo<X><Y><Z>\22\00",
// CHECK: private unnamed_addr constant [13 x i8] c"{Intf=@@@@#}\00",

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

int main()
{
	const char * en = @encode(Intf);
}
