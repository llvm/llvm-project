// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-address-of-temporary -Did="void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// rdar:// 8570020

@interface Foo {
	Foo *foo;
}

@property (retain, nonatomic) Foo *foo;

@end

@implementation Foo

- (Foo *)foo {
    if (!foo) {
        foo = 0;
    }
    return foo;
}


- (void) setFoo : (Foo *) arg {
  foo = arg;
}

@synthesize foo;

@end 

