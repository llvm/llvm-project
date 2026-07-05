// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -fblocks -Wno-address-of-temporary -D"SEL=void*" -U__declspec -D"__declspec(X)=" %t-rw.cpp
// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-modern-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -Werror -Wno-address-of-temporary -D"SEL=void*" -U__declspec  -D"__declspec(X)=" %t-modern-rw.cpp

typedef unsigned long size_t;

void f(void (^block)(void));

@interface X {
        int y;
}
- (void)foo;
@end

@implementation X
- (void)foo {
        __block int kerfluffle;
        __block int x;
        f(^{
                f(^{
                                y = 42;
                            kerfluffle = 1;
			    x = 2;
                });
        });
}
@end
