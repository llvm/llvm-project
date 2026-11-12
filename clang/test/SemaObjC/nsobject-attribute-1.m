// RUN: %clang_cc1 -fblocks -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics

@interface NSObject
- (id)self;
- (id)copy;
@end

typedef struct _foo  *__attribute__((NSObject)) Foo_ref;

@interface TestObject {
    Foo_ref dict;
}
@property(retain) Foo_ref dict;
@end

@implementation TestObject
@synthesize dict;
@end

@interface NSDictionary
- (int)retainCount;
@end

int main(int argc, char *argv[]) {
    NSDictionary *dictRef;
    Foo_ref foo = (Foo_ref)dictRef;

    // do Properties retain?
    int before = [dictRef retainCount];
    int after = [dictRef retainCount];

    if ([foo retainCount] != [dictRef retainCount]) {
    }

    // do Blocks retain?
    {
        void (^block)(void) = ^{
            [foo self];
        };
        before = [foo retainCount];
        id save = [block copy];
        after = [foo retainCount];
        if (after <= before) {
            ;
        }
    }
    return 0;
}
