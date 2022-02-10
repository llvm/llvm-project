// RUN: %clang_cc1 %s -fsyntax-only -verify

__attribute__((__objc_exception__))
@interface NSException {
  int x;
}

@end


__attribute__((__objc_exception__)) // expected-error {{'__objc_exception__' attribute only applies to Objective-C interfaces}}
int X;

__attribute__((__objc_exception__)) // expected-error {{'__objc_exception__' attribute only applies to Objective-C interfaces}}
void foo(void);

