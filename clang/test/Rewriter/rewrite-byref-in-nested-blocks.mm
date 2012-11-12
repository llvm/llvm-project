// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-address-of-temporary -D"SEL=void*" -U__declspec -D"__declspec(X)=" %t-rw.cpp
// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-modern-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Werror -Wno-address-of-temporary -D"SEL=void*" -U__declspec  -D"__declspec(X)=" %t-modern-rw.cpp
// radar 7692350

// rdar://11375908
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
        // radar 7692183
        __block x; 
        f(^{
                f(^{
                                y = 42;
                            kerfluffle = 1;
			    x = 2;
                });
        });
}
@end
