// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface SomeClass  @end

int fn1(SomeClass *obj) {
        obj->privateIvar = 1; // expected-error {{'SomeClass' does not have a member named 'privateIvar}}
        return obj->publicIvar;     // expected-error {{'SomeClass' does not have a member named 'publicIvar'}}
}

@interface SomeClass () {
// @private by default
        int privateIvar;
@public
        int publicIvar;
}
@end

int fn2(SomeClass *obj) {
	obj->publicIvar = 1;
        return obj->publicIvar    // ok
             + obj->privateIvar;  // expected-error {{instance variable 'privateIvar' is private}}
}

@implementation SomeClass

int fn3(SomeClass *obj) {
	obj->privateIvar = 2;
        return obj->publicIvar    // ok
             + obj->privateIvar;  // ok
    }
@end

@interface SomeClass (Category)
    {	
        int categoryIvar; // expected-error {{instance variables may not be placed in categories}}
    }
@end

@interface SomeClass (Category1)
    {	
    }
@end
