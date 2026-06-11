// RUN: %clang_cc1                  -Wpedantic %s -fsyntax-only -isystem %S/Inputs -verify=ext
// RUN: %clang_cc1 -std=c2y         -Wpedantic %s -fsyntax-only -isystem %S/Inputs -verify
// RUN: %clang_cc1 -std=c2y -Wpre-c2y-compat   %s -fsyntax-only -isystem %S/Inputs -verify=pre
// RUN: %clang_cc1                            -pedantic %s -fsyntax-only -isystem %S/Inputs -verify=ext
// RUN: %clang_cc1 -std=c2y -Wpre-c2y-compat  -pedantic %s -fsyntax-only -isystem %S/Inputs -verify=pre
// RUN: %clang_cc1                            -pedantic-errors %s -fsyntax-only -isystem %S/Inputs -verify=pedant
// RUN: %clang_cc1 -std=c2y -Wpre-c2y-compat  -pedantic-errors %s -fsyntax-only -isystem %S/Inputs -verify=pre

#include <__counter__-system-header.h>

// expected-no-diagnostics

int tu_direct_reference = __COUNTER__; // #errorline
// ext-warning@#errorline {{'__COUNTER__' is a C2y extension}}
// pre-warning@#errorline {{'__COUNTER__' is incompatible with standards before C2y}}
// pedant-error@#errorline {{'__COUNTER__' is a C2y extension}}
int tu_counter_alias = COUNTER_ALIAS;
int tu_counter_macro = COUNTER_MACRO();
