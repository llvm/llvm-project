// RUN: not %clang_cc1_only -c %s -o - > /dev/null
// PR 1603
void func()
{
   const int *arr;
   arr[0] = 1;  // expected-error {{assignment of read-only location}}
}

struct foo {
  int bar;
};
struct foo sfoo = { 0 };

int func2()
{
  const struct foo *fp;
  fp = &sfoo;
  fp[0].bar = 1;  // expected-error {{ assignment of read-only member}}
  return sfoo.bar;
}
