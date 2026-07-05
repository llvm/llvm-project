// RUN: %clang_cc1 -fsyntax-only -verify -triple %ms_abi_triple -Wno-objc-root-class %s
// expected-no-diagnostics

class Foo {
  ~Foo();
};

@interface bar
- (void) my_method: (Foo)arg;
@end

@implementation bar
- (void) my_method: (Foo)arg { // no error; MS ABI will call Foo's dtor, but we skip the access check.
}
@end
