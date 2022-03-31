// RUN: %clang_cc1 %s -verify -fsyntax-only
// expected-no-diagnostics

@class NSString;

void c1(id *a);

void t1(void)
{
  NSString *s __attribute((cleanup(c1)));
}
