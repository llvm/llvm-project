// RUN: %clang_cc1 -fsyntax-only -Werror -verify -Wno-objc-root-class %s
// expected-no-diagnostics

@interface MyClass {
    const char	*_myName;
}

@property const char *myName;

- (const char *)myName;
- (void)setMyName:(const char *)name;

@end

@implementation MyClass

@synthesize myName = _myName;

@end
