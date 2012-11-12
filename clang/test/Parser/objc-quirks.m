// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s

// FIXME: This is a horrible error message here. Fix.
int @"s" = 5;  // expected-error {{prefix attribute must be}}


// rdar://6480479
@interface A // expected-note {{class started here}}
}; // expected-error {{missing '@end'}} \
// expected-error {{extraneous closing brace ('}')}} \
// expected-warning{{extra ';' outside of a function}}




// PR6811
// 'super' isn't an expression, it is a magic context-sensitive keyword.
@interface A2 {
  id isa;
}
- (void)a;
@end

@interface B2 : A2 @end
@implementation B2
- (void)a
{
  [(super) a];  // expected-error {{use of undeclared identifier 'super'}}
}
@end

@compatibility_alias A3 A2;
