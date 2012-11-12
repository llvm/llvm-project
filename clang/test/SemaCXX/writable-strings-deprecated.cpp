// RUN: %clang_cc1 -fsyntax-only -Wno-deprecated-writable-strings -verify %s
// RUN: %clang_cc1 -fsyntax-only -fwritable-strings -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-write-strings -verify %s
// rdar://8827606

char *fun(void)
{
   return "foo";
}

void test(bool b)
{
  ++b; // expected-warning {{incrementing expression of type bool is deprecated}}
}
