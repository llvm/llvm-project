// RUN: %clang_cc1  -fsyntax-only -verify %s
// rdar://9092208

__attribute__((unavailable("not available")))
@interface MyClass { // expected-note 8 {{declaration has been explicitly marked unavailable here}}
@public
    void *_test;
    MyClass *ivar; // no error.
}

- (id)self;
- new;
+ (void)addObject:(id)anObject;
- (MyClass *)meth; // no error.

@end

@interface Foo {
  MyClass *ivar; // expected-error {{unavailable}}
}
- (MyClass *)meth; // expected-error {{unavailable}}
@end

@interface MyClass (Cat1)
- (MyClass *)meth; // no error.
@end

@interface MyClass (Cat2) // no error.
@end

@implementation MyClass (Cat2) // expected-error {{unavailable}}
@end

int main() {
 [MyClass new]; // expected-error {{'MyClass' is unavailable: not available}}
 [MyClass self]; // expected-error {{'MyClass' is unavailable: not available}}
 [MyClass addObject:((void *)0)]; // expected-error {{'MyClass' is unavailable: not available}}

 MyClass *foo = [MyClass new]; // expected-error 2 {{'MyClass' is unavailable: not available}}

 return 0;
}
