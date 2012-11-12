// RUN: %clang_cc1 %s -fsyntax-only -verify

__attribute__((__objc_exception__))
@interface NSException {
  int x;
}

@end


__attribute__((__objc_exception__)) // expected-error {{attribute may only be applied to an Objective-C interface}}
int X;

__attribute__((__objc_exception__)) // expected-error {{attribute may only be applied to an Objective-C interface}}
void foo();

