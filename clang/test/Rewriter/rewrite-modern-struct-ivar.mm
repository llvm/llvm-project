// RUN: %clang_cc1 -E %s -o %t.mm
// RUN: %clang_cc1 -fblocks -rewrite-objc -fms-extensions %t.mm -o %t-rw.cpp 
// RUN: FileCheck --input-file=%t-rw.cpp %s
// RUN: %clang_cc1 -fsyntax-only -Werror -Wno-address-of-temporary -Wno-c++11-narrowing -std=c++11 -D"Class=void*" -D"id=void*" -D"SEL=void*" -U__declspec -D"__declspec(X)=" %t-rw.cpp

struct S {
    int i1;
    double d1;
    void (^block1)();
};

@interface I
{
  struct S struct_ivar;

  struct S *pstruct_ivar;
}
@end

@implementation I
- (struct S) dMeth{ return struct_ivar; }
@end

// CHECK: return (*(struct S *)((char *)self + OBJC_IVAR_$_I$struct_ivar));

// rdar://11323187
@interface Foo{
    @protected 
    struct {
        int x:1;
        int y:1;
    } bar;

    struct _S {
        int x:1;
        int y:1;
    } s;

}
@end
@implementation Foo
- (void)x {
  bar.x = 0;
  bar.y = 1;

  s.x = 0;
  s.y = 1;
}
@end

// CHECK: (*(decltype(((Foo_IMPL *)0U)->bar) *)((char *)self + OBJC_IVAR_$_Foo$bar)).x = 0;
// CHECK: (*(struct _S *)((char *)self + OBJC_IVAR_$_Foo$s)).x = 0;
