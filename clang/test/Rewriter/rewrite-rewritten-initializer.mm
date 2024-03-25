// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D"SEL=void*" -U__declspec -D"__declspec(X)=" %t-rw.cpp
// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-rw-modern.cpp
// RUN: %clang_cc1 -fsyntax-only -Werror -Wno-address-of-temporary -D"SEL=void*" -U__declspec -D"__declspec(X)=" %t-rw-modern.cpp

typedef unsigned long size_t;
typedef void * id;
void *sel_registerName(const char *);

@interface NSMutableString
- (NSMutableString *)string;
@end

@interface Z
@end

@implementation Z

- (void)x {
        id numbers;
    int i, numbersCount = 42;
    __attribute__((__blocks__(byref))) int blockSum = 0;
    void (^add)(id n, int idx, char *stop) = ^(id n, int idx, char *stop) { };
    [numbers enumerateObjectsUsingBlock:add];
    NSMutableString *forwardAppend = [NSMutableString string];
    __attribute__((__blocks__(byref))) NSMutableString *blockAppend = [NSMutableString string];
}

@end

