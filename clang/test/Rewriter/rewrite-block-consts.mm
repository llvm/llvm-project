// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D__block="" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// rdar:// 8243071

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
