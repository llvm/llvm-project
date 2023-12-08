// RUN: %clang_cc1 -fsyntax-only -verify -fms-extensions -triple i386-pc-win32 %s

wchar_t f();
__wchar_t f(); // No error, wchar_t and __wchar_t are the same type.

__wchar_t g = L'a';
__wchar_t s[] = L"Hello world!";

unsigned short t[] = L"Hello world!"; // expected-error{{array initializer must be an initializer list}}

wchar_t u[] = 1; // expected-error{{array initializer must be an initializer list or wide string literal}}
__wchar_t v[] = 1; // expected-error{{array initializer must be an initializer list or wide string literal}}
