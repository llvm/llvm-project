// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -Wno-address-of-temporary -Did="void *" -D"SEL=void*" -D"__declspec(X)=" -emit-llvm -o %t %t-rw.cpp
// radar 7946975

void *sel_registerName(const char *);

@interface foo
@end

@interface foo2 : foo
+ (id)x;
@end

typedef void (^b_t)(void);

void bar(b_t block);

void f() {
        static id foo = 0;
        bar(^{
                foo = [foo2 x];
        });
}

