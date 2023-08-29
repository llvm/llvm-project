// RUN: %clang_cc1 -fsyntax-only -verify %s

// expected-note@+1 {{previous definition is here}}
@interface blah { // expected-note {{class started here}}
    @private
}
// since I forgot the @end here it should say something

// expected-error@+1 {{duplicate interface definition for class 'blah'}}
@interface blah  // expected-error {{missing '@end'}}
@end

