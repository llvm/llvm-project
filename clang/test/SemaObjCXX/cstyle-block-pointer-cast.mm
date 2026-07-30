// RUN: %clang_cc1 -fsyntax-only -verify -fblocks -Wno-objc-root-class %s

typedef int (^blocktype)(int a, int b);

@interface A {
    A* a;
    id b;
    Class c;
}
- (blocktype)Meth;
@end

@implementation A
- (blocktype)Meth {
        if (b)
	  return (blocktype)b;
        else if (a)
          return (blocktype)a; // expected-error {{C-style cast from 'A *' to 'blocktype' (aka 'int (^)(int, int)') is not allowed}}
        else
          return (blocktype)c;
}
@end

@interface B {
    blocktype a;
    blocktype b;
    blocktype c;
}
- (id)Meth;
@end

@implementation B
- (id)Meth {
        if (a)
          return (A*)a; // expected-error {{C-style cast from 'blocktype' (aka 'int (^)(int, int)') to 'A *' is not allowed}}
        if (b)
	  return (id)b;
        if (c)
	  return (Class)b;
}
@end
