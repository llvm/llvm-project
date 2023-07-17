// RUN: %clang_cc1 -x objective-c -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o %t-rw.cpp
// RUN: %clang_cc1 -Wno-address-of-temporary -Did="void *" -D"SEL=void*" -D"__declspec(X)=" -emit-llvm -o %t %t-rw.cpp
// radar 7987817

void *sel_registerName(const char *);

@interface Test {
}
@end

@implementation Test

- (void)enumerateProvidersWithBlock:(void (^)(void))block {
    block();
}

- (void)providerEnumerator {
    ^(void (^providerBlock)(void)) {
        [self enumerateProvidersWithBlock:providerBlock];
    };
}

- (void)testNilBlock {
    [self enumerateProvidersWithBlock:0];
}

@end



int main(int argc, char *argv[]) {
    return 0;
}
