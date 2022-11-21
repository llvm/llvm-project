// RUN: %clang_cc1 -fsyntax-only -verify=gnu -std=gnu17 %s
// RUN: %clang_cc1 -fsyntax-only -verify=gnu -std=gnu2x %s
// RUN: %clang_cc1 -fsyntax-only -verify=std -std=c17 %s
// RUN: %clang_cc1 -fsyntax-only -verify=std -std=c2x %s

// std-no-diagnostics

// 'index' is a builtin library function, but only in GNU mode. So this should
// give an error in GNU modes but be okay in non-GNU mode.
// FIXME: the error is correct, but these notes are pretty awful.
int index; // gnu-error {{redefinition of 'index' as different kind of symbol}} \
              gnu-note {{unguarded header; consider using #ifdef guards or #pragma once}} \
              gnu-note {{previous definition is here}}
