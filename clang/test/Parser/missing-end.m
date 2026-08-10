// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface AAA // expected-note {{class started here}}
{
}
@ x// expected-error{{expected an Objective-C directive after '@'}}
// expected-error{{missing '@end'}}
