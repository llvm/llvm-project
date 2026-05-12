// RUN: %clang_cc1 -fblocks -fsyntax-only -verify %s

@interface MyClass {
    // expected-warning@-1 {{class 'MyClass' defined without specifying a base class}}
    // expected-note@-2 {{add a super class to fix this problem}}
    __block int _myIvar;
}
@end

@implementation MyClass
@end
