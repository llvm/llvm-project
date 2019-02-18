// RUN: %clang_cc1 -fsyntax-only %s
// RUN: %clang_cc1 -ast-print %s -o %t.1.c
// RUN: %clang_cc1 -ast-print %t.1.c -o %t.2.c
// RUN: diff %t.1.c %t.2.c
// RUN: %clang_cc1 -ast-dump %s
// RUN: %clang_cc1 -ast-dump-all %s

#include "c-language-features.inc"
