// RUN: %clang_cc1 -triple=i686-apple-darwin9 -emit-llvm -o %t %s
// RUN: grep -e "T@\\\22<X>\\\22" %t
// RUN: grep -e "T@\\\22<X><Y>\\\22" %t
// RUN: grep -e "T@\\\22<X><Y><Z>\\\22" %t
// RUN: grep -e "T@\\\22Foo<X><Y><Z>\\\22" %t

@protocol X, Y, Z;
@class Foo;

@protocol Proto
@property (copy) id <X> x;
@property (copy) id <X, Y> xy;
@property (copy) id <X, Y, Z> xyz;
@property(copy)  Foo <X, Y, Z> *fooxyz;
@end

@interface Intf <Proto>
{
id <X> IVAR_x;
id <X, Y> IVAR_xy;
id <X, Y, Z> IVAR_xyz;
Foo <X, Y, Z> *IVAR_Fooxyz;
}
@end

@implementation Intf 
@dynamic x, xy, xyz, fooxyz;
@end

/**
This protocol should generate the following metadata:
struct objc_property_list __Protocol_Test_metadata = {
  sizeof(struct objc_property), 4,
  {
    { "x", "T@\"<X>\"" },
    { "xy", "T@\"<X><Y>\"" },
    { "xyz", "T@\"<X><Y><Z>\"" },
    { "fooxyz", "T@\"Foo<X><Y><Z>\"" }
  }
};

"T@\"<X><Y><Z>\",D
*/
