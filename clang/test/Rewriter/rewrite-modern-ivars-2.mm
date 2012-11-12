// RUN: %clang_cc1 -triple i386-apple-darwin9 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -triple i386-apple-darwin9 -fsyntax-only -fblocks -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

@interface B @end

@interface A {
  struct s0 {
    int f0;
    int f1;
  } f0;
  id f1;
__weak B *f2;
  int f3 : 5;
  struct s1 {
    int *f0;
    int *f1;
  } f4[2][1];
}
@end

@interface C : A
@property int p3;
@end

@implementation C
@synthesize p3 = _p3;
@end

@interface A()
@property int p0;
@property (assign) __strong id p1;
@property (assign) __weak id p2;
@end

// FIXME: Check layout for this class, once it is clear what the right
// answer is.
@implementation A
@synthesize p0 = _p0;
@synthesize p1 = _p1;
@synthesize p2 = _p2;
@end

@interface D : A
@property int p3;
@end

// FIXME: Check layout for this class, once it is clear what the right
// answer is.
@implementation D
@synthesize p3 = _p3;
@end

typedef unsigned short UInt16;


typedef signed char BOOL;
typedef unsigned int FSCatalogInfoBitmap;

@interface NSFileLocationComponent {
    @private

    id _specifierOrStandardizedPath;
    BOOL _carbonCatalogInfoAndNameAreValid;
    FSCatalogInfoBitmap _carbonCatalogInfoMask;
    id _name;
    id _containerComponent;
    id _presentableName;
    id _iconAsAttributedString;
}
@end

@implementation NSFileLocationComponent @end

// rdar://11229770

@interface Foo {
  int bar:26;
}
@end

@implementation Foo
@end

@interface Foo1 {
  int bar:26;
  int bar2:4;
}
@end

@implementation Foo1
@end

@interface Foo3 {
  int foo;
  int bar:26;
}
@end

@implementation Foo3
@end

