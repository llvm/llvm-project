// RUN: %clang_cc1 -fsyntax-only -verify %s -triple powerpc-ibm-aix
// RUN: %clang_cc1 -fsyntax-only -verify %s -triple powerpc64-ibm-aix
// expected-no-diagnostics

extern __builtin_va_list ap;
extern char *ap;
