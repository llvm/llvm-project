// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp 
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -fblocks -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

@interface F {
  int supervar;
}
@end

@interface G : F {
@public
  int ivar;
}
@end

@implementation G
- (void)foo:(F *)arg {
        int q = arg->supervar;
        int v = ((G *)arg)->ivar;
}
@end

void objc_assign_strongCast(id);
void __CFAssignWithWriteBarrier(void **location, void *value) {
        objc_assign_strongCast((id)value);
}

// radar 7607605
@interface RealClass {
        @public
        int f;
}
@end

@implementation RealClass
@end

@interface Foo {
        id reserved;
}
@end

@implementation Foo
- (void)bar {
        ((RealClass*)reserved)->f = 99;
}
@end
