// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar://8283484
@interface blah { // expected-note {{class started here}}
    @private
}
// since I forgot the @end here it should say something

@interface blah  // expected-error {{missing '@end'}}
@end // and Unknown type name 'end' here

