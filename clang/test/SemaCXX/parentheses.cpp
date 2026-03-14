// RUN: %clang_cc1 -verify -Wlogical-op-parentheses %s

// PR16930, PR16727:
template<class Foo>
bool test(Foo f, int *array)
{
  return false && false || array[f.get()]; // expected-warning {{'&&' within '||'}} expected-note {{parentheses}}
}
