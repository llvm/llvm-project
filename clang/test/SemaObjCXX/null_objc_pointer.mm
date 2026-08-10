// RUN: %clang_cc1 -fsyntax-only -verify -Wnull-arithmetic %s
#define NULL __null

@interface X
@end

void f() {
  bool b;
  X *d;
  b = d < NULL || NULL < d || d > NULL || NULL > d; // expected-error 4{{ordered comparison between pointer and zero}}
  b = d <= NULL || NULL <= d || d >= NULL || NULL >= d; // expected-error 4{{ordered comparison between pointer and zero}}
  b = d == NULL || NULL == d || d != NULL || NULL != d;
}
