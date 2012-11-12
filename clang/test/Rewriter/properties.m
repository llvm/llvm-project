// RUN: %clang_cc1 -x objective-c -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fms-extensions -Wno-address-of-temporary -Did="void *" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

void *sel_registerName(const char *);

@interface Foo {
    int i;
    int rrrr;
    Foo *o;
}
@property int i;
@property(readonly) int rrrr;
@property int d;
@property(retain) Foo *o;

- (void)foo;
@end

@implementation Foo
@synthesize i;
@synthesize rrrr;
@synthesize o;

@dynamic d;

- (void)foo {
    i = 99;
}

- (int)bar {
  return i;
}
@end

@interface Bar {
}
@end

@implementation Bar

static int func(int i) { return 0; }

- (void)baz {
    Foo *obj1, *obj2;
    int i;
    if (obj1.i == obj2.rrrr)
      obj1.i = 33;
    obj1.i = func(obj2.rrrr);
    obj1.i = obj2.rrrr;
    obj1.i = (obj2.rrrr);
    [obj1 setI:[obj2 rrrr]];
    obj1.i = [obj2 rrrr];
    obj1.i = 3 + [obj2 rrrr];
    i = obj1.o.i;
    obj1.o.i = 77;
}
@end
