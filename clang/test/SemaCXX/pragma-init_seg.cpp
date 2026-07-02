// RUN: %clang_cc1 -fsyntax-only -verify -fms-extensions %s -triple x86_64-pc-win32
// RUN: %clang_cc1 -fsyntax-only -verify -fms-extensions -std=c++98 %s -triple x86_64-pc-win32
// RUN: %clang_cc1 -fsyntax-only -verify -fms-extensions -std=c++11 %s -triple x86_64-pc-win32
// RUN: %clang_cc1 -fsyntax-only -verify -fms-extensions %s -triple i386-apple-darwin13.3.0
// RUN: %clang_cc1 -fsyntax-only -verify -fms-extensions -std=c++98 %s -triple i386-apple-darwin13.3.0
// RUN: %clang_cc1 -fsyntax-only -verify -fms-extensions -std=c++11 %s -triple i386-apple-darwin13.3.0

#ifndef __APPLE__
#pragma init_seg(L".my_seg") // expected-warning {{expected 'compiler', 'lib', 'user', or a string literal}}
#pragma init_seg( // expected-warning {{expected 'compiler', 'lib', 'user', or a string literal}}
#pragma init_seg asdf // expected-warning {{missing '('}}
#pragma init_seg) // expected-warning {{missing '('}}
#pragma init_seg("a" "b") // no warning
#pragma init_seg("a", "b") // expected-warning {{expected identifier in '#pragma init_seg'}}
#pragma init_seg("a") asdf // expected-warning {{extra tokens at end of '#pragma init_seg'}}
#pragma init_seg("\x") // expected-error {{\x used with no following hex digits}}
#pragma init_seg("a" L"b") // expected-warning {{expected non-wide string literal in '#pragma init_seg'}}

#pragma init_seg(compiler)

int __cdecl myexit(void (__cdecl *pf)(void));
#pragma init_seg(".myseg", myexit)

int bad_func(int x);
#pragma init_seg(".myseg2", bad_func) // expected-error {{'#pragma init_seg' function 'bad_func' must have type 'int (*)(void (__cdecl *)(void))'}}

#pragma init_seg(".myseg3", no_such_func) // expected-error {{use of undeclared identifier 'no_such_func'}}

#else
#pragma init_seg(compiler) // expected-warning {{'#pragma init_seg' is only supported when targeting a Microsoft environment}}
#endif

int f();
int __declspec(thread) x = f(); // expected-error {{initializer for thread-local variable must be a constant expression}}
#if __cplusplus >= 201103L
// expected-note@-2 {{use 'thread_local' to allow this}}
#endif
