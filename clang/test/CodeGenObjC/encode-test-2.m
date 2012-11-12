// RUN: %clang_cc1 -triple=i686-apple-darwin9 -emit-llvm -o %t %s
// RUN: grep -e "@\\\22<X>\\\22" %t
// RUN: grep -e "@\\\22<X><Y>\\\22" %t
// RUN: grep -e "@\\\22<X><Y><Z>\\\22" %t
// RUN: grep -e "@\\\22Foo<X><Y><Z>\\\22" %t
// RUN: grep -e "{Intf=@@@@#}" %t  

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
