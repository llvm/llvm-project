// RUN: %clang_cc1 %s -fsyntax-only -Wno-unused-value -Wmicrosoft -verify -fms-compatibility -triple i386-pc-win32
int __stdcall f(void); /* expected-note {{previous declaration is here}} */

int __cdecl f(void) { /* expected-error {{function declared 'cdecl' here was previously declared 'stdcall'}} */
  return 0;
}
