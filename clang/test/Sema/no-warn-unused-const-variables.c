// RUN: %clang_cc1 -fsyntax-only -Wunused-const-variable -x c-header -ffreestanding -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wunused-const-variable -x c++-header -ffreestanding -verify %s
// expected-no-diagnostics
static const int unused[] = { 2, 3, 5, 7, 11, 13 };
