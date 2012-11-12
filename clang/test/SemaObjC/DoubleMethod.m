// RUN: %clang_cc1 -Wduplicate-method-match -fsyntax-only -verify -Wno-objc-root-class %s

@interface Subclass
{
    int ivar;
}

- (void) method; // expected-note {{previous declaration is here}}
- (void) method; // expected-warning {{multiple declarations of method 'method' found and ignored}}
@end

@implementation Subclass
- (void) method {;} // expected-note {{previous declaration is here}}
- (void) method {;} // expected-error {{duplicate declaration of method 'method'}}
@end

int main (void) {
    return 0;
}
