// RUN: %clang_cc1 -fsyntax-only -verify %std_cxx14- %s -DWARNING
// RUN: %clang_cc1 -fsyntax-only -std=c++98 -verify %s -DWARNING
// RUN: %clang_cc1 -fsyntax-only -std=c++98 -Wno-deprecated-writable-strings -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++98 -Wno-deprecated -Wdeprecated-increment-bool -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++98 -fwritable-strings -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++98 -Wno-write-strings -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++98 -Werror=c++11-compat -verify %s -DERROR
// RUN: %clang_cc1 -fsyntax-only -std=c++98 -Werror=deprecated -Wno-error=deprecated-increment-bool -verify %s -DERROR
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s -DWARNING
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s -Wno-deprecated -Wdeprecated-increment-bool -DWARNING
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s -pedantic-errors -DERROR

char *fun(void)
{
   return "foo";
#if defined(ERROR)
#if __cplusplus >= 201103L
   // expected-error@-3 {{ISO C++11 does not allow conversion from string literal to 'char *'}}
#else
   // expected-error@-5 {{conversion from string literal to 'char *' is deprecated}}
#endif
#elif defined(WARNING)
#if __cplusplus >= 201103L
   // expected-warning@-9 {{ISO C++11 does not allow conversion from string literal to 'char *'}}
#else
   // expected-warning@-11 {{conversion from string literal to 'char *' is deprecated}}
#endif
#endif
}

#if __cplusplus < 201703L
void test(bool b)
{
  ++b; // expected-warning {{incrementing expression of type bool is deprecated}}
}
#endif
