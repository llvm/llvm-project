// RUN: %clang_cc1 -fsyntax-only -Wno-deprecated-writable-strings -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-deprecated -Wdeprecated-increment-bool -verify %s
// RUN: %clang_cc1 -fsyntax-only -fwritable-strings -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-write-strings -verify %s
// RUN: %clang_cc1 -fsyntax-only -Werror=c++11-compat -verify %s -DERROR
// RUN: %clang_cc1 -fsyntax-only -Werror=deprecated -Wno-error=deprecated-increment-bool -verify %s -DERROR
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s -Wno-deprecated -Wdeprecated-increment-bool
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s -pedantic-errors -DERROR
// rdar://8827606

char *fun(void)
{
   return "foo";
#if __cplusplus >= 201103L
#ifdef ERROR
   // expected-error@-3 {{ISO C++11 does not allow conversion from string literal to 'char *'}}
#else
   // expected-warning@-5 {{ISO C++11 does not allow conversion from string literal to 'char *'}}
#endif
#elif defined(ERROR)
   // expected-error@-8 {{deprecated}}
#endif
}

void test(bool b)
{
  ++b; // expected-warning {{incrementing expression of type bool is deprecated}}
}
