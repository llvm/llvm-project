// RUN: %clang_cc1 -triple i686-unknown-windows-msvc -fsyntax-only -fdeclspec -verify %s
// RUN: %clang_cc1 -triple thumbv7-unknown-windows-msvc -fsyntax-only -fdeclspec -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fsyntax-only -fdeclspec -verify %s
#if defined(_M_IX86) || defined(_M_ARM)
// CHECK: expected-no-diagnostics
#endif

void __declspec(naked) f(void) {}
#if !defined(_M_IX86) && !defined(_M_ARM)
// expected-error@-2{{'naked' attribute is not supported on 'x86_64'}}
#endif
