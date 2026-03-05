// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple systemz -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple powerpc64-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -DEXPOK -triple powerpc-ibm-aix -fsyntax-only -verify %s

#ifdef EXPOK
#pragma comment(copyright,"copyright") // expected-no-diagnostics
#else
#pragma comment(copyright,"copyright") // expected-warning {{'#pragma comment copyright' ignored}}
#endif
