// RUN: %clang_cc1 -triple i386-pc-win32 -fsyntax-only -fno-wchar -verify %s
wchar_t x; // expected-error {{unknown type name 'wchar_t'}}

typedef unsigned short wchar_t;
void foo(const wchar_t* x);

void bar() {
  foo(L"wide string literal");
}
