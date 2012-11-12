// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-address-of-temporary -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// rdar : // 8213998

typedef unsigned int NSUInteger;

typedef struct _NSRange {
    NSUInteger location;
    NSUInteger length;
} NSRange;

static __inline NSRange NSMakeRange(NSUInteger loc, NSUInteger len) {
    NSRange r;
    r.location = loc;
    r.length = len;
    return r;
}

void bar() {
    __block NSRange previousRange = NSMakeRange(0, 0);    
    void (^blk)() = ^{
        previousRange = NSMakeRange(1, 0);
    };
}
