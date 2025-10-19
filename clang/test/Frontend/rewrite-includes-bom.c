// RUN: cat %S/Inputs/rewrite-includes-bom.h | od -t x1 | grep -q 'ef\s*bb\s*bf'
// RUN: %clang_cc1 -E -frewrite-includes -I %S/Inputs %s -o %t.c
// RUN: cat %t.c | od -t x1 | not grep -q 'ef\s*bb\s*bf'
// RUN: %clang_cc1 -fsyntax-only -verify %t.c
// expected-no-diagnostics

#include "rewrite-includes-bom.h"
