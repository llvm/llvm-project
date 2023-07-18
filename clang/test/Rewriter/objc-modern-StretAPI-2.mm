// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// rdar://12142241

extern "C" void *sel_registerName(const char *);
typedef unsigned long size_t;

typedef unsigned long NSUInteger;
typedef struct _NSRange {
    NSUInteger location;
    NSUInteger length;
} NSRange;


@interface NSIndexSet
- (NSRange)rangeAtIndex:(NSUInteger)rangeIndex;
@end

@interface NSArray
@end

@implementation NSArray
- (NSArray *)objectsAtIndexes:(NSIndexSet *)iset {

    NSUInteger ridx = 0;
    NSRange range = [iset rangeAtIndex:ridx];
    return 0;
}
@end

