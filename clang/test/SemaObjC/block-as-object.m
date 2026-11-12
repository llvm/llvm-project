// RUN: %clang_cc1 %s -fsyntax-only -verify -fblocks
// expected-no-diagnostics

@interface Whatever
- copy;
@end

typedef long (^MyBlock)(id obj1, id obj2);

void foo(MyBlock b) {
    id bar = [b copy];
}

void foo2(id b) {
}

void foo3(void (^block)(void)) {
    foo2(block);
    id x;
    foo(x);
}
