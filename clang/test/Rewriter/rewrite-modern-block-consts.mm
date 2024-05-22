// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-modern-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Werror -Wno-address-of-temporary -U__declspec -D"__declspec(X)=" %t-modern-rw.cpp

typedef unsigned long size_t;

void x(int y) {}
void f() {
    const int bar = 3;
    int baz = 4;
    __block int bab = 4;
    __block const int bas = 5;
    void (^b)() = ^{
        x(bar);
        x(baz);
        x(bab);
        x(bas);
        b();
    };
    b();
} 
