// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
@interface NSObject
- (void)doSomething __attribute__((nodebug));
@end
