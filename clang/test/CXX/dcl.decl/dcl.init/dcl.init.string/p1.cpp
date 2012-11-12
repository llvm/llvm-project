// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

char x1[]("hello");
extern char x1[6];

char x2[] = "hello";
extern char x2[6];

char x3[] = { "hello" };
extern char x3[6];

wchar_t x4[](L"hello");
extern wchar_t x4[6];

wchar_t x5[] = L"hello";
extern wchar_t x5[6];

wchar_t x6[] = { L"hello" };
extern wchar_t x6[6];
