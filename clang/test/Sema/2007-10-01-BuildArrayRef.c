// RUN: %clang_cc1 -fsyntax-only -verify %s
// PR 1603
void func(void)
{
   const int *arr;
   arr[0] = 1;  // expected-error {{read-only variable is not assignable}}
}

struct foo {
  int bar;
};
struct foo sfoo = { 0 };

int func2(void)
{
  const struct foo *fp;
  fp = &sfoo;
  fp[0].bar = 1;  // expected-error {{read-only variable is not assignable}}
  return sfoo.bar;
}
