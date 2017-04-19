// RUN: %clang_cc1 -fsyntax-only -verify %s

@protocol NSObject
@end

@protocol <#protocol name#> <NSObject> // expected-error {{editor placeholder in source file}}
// expected-note@-1 {{protocol started here}}

// FIXME: We could do much better with this, if we recognized
// placeholders somehow. However, we're content with not generating
// bogus 'archaic' warnings with bad location info.
<#methods#> // expected-error {{editor placeholder in source file}}

@end // expected-error {{prefix attribute must be followed by an interface or protocol}} expected-error {{missing '@end'}}
