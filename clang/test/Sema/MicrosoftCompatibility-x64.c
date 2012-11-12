// RUN: %clang_cc1 %s -fsyntax-only -Wno-unused-value -Wmicrosoft -verify -fms-compatibility -triple x86_64-pc-win32
int __stdcall f(void); /* expected-warning {{calling convention '__stdcall' ignored for this target}} */

/* This should compile without warning because __stdcall is treated
as __cdecl in MS compatibility mode for x64 compiles*/
int __cdecl f(void) {
  return 0;
}
