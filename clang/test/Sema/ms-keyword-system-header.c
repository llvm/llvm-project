// RUN: %clang_cc1 -fms-extensions -D MS -isystem %S/Inputs %s -fsyntax-only -verify
// RUN: %clang_cc1 -fms-extensions -D MS -Wno-keyword-compat -I %S/Inputs %s -fsyntax-only -verify
// RUN: %clang_cc1 -fms-extensions -D MS -D NOT_SYSTEM -I %S/Inputs %s -fsyntax-only -verify
// RUN: %clang_cc1 -isystem %S/Inputs %s -fsyntax-only -verify

// PR17824: GNU libc uses MS keyword __uptr as an identifier in C mode
#include <ms-keyword-system-header.h>

void fn(void) {
  WS ws;
  ws.__uptr = 0;
#ifdef MS
  // expected-error@-2 {{expected identifier}}
#else
  // expected-no-diagnostics
#endif
}
