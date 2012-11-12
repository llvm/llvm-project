// RUN: %clang_cc1 -fsyntax-only -verify %s

// FIXME: We could do much better with this, if we recognized
// placeholders somehow. However, we're content with not generating
// bogus 'archaic' warnings with bad location info.
@protocol <#protocol name#> <NSObject> // expected-error 2{{expected identifier}} \
// expected-error{{cannot find protocol declaration for 'NSObject'}} \
// expected-warning{{protocol qualifiers without 'id'}}

<#methods#>  // expected-error{{expected identifier}}

@end // expected-error{{prefix attribute}}
