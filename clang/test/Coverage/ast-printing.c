// RUN: %clang_cc1 -fsyntax-only -Wno-strict-prototypes %s
// RUN: %clang_cc1 -ast-print -Wno-strict-prototypes %s -o %t.1.c
// RUN: %clang_cc1 -ast-print -Wno-strict-prototypes %t.1.c -o %t.2.c
// RUN: diff %t.1.c %t.2.c
// RUN: %clang_cc1 -ast-dump -Wno-strict-prototypes %s
// RUN: %clang_cc1 -ast-dump-all -Wno-strict-prototypes %s

#include "c-language-features.inc"
