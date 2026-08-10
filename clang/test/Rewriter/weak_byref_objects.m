// RUN: %clang_cc1 -fblocks -triple i386-apple-darwin9 -fobjc-gc -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o -

#define nil 0
int main(void) {
        __weak __block id foo = nil;
        __block id foo2 = nil;
        id foo3 = nil;

        void (^myblock)(void) = ^{
                foo = nil;
                foo2 = nil;
                [foo3 bar];
                id foo4 = foo3;
        };
}
