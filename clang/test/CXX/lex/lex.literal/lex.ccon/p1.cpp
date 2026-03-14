// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

// Check types of char literals
extern char a;
extern __typeof('a') a;
extern int b;
extern __typeof('asdf') b;
extern wchar_t c;
extern __typeof(L'a') c;
#if __cplusplus >= 201103L
extern char16_t d;
extern __typeof(u'a') d;
extern char32_t e;
extern __typeof(U'a') e;
#endif
