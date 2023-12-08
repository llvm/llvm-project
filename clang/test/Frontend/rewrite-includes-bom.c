// RUN: grep -q $'^\xEF\xBB\xBF' %S/Inputs/rewrite-includes-bom.h
// RUN: %clang_cc1 -E -frewrite-includes -I %S/Inputs %s -o %t.c
// RUN: ! grep -q $'\xEF\xBB\xBF' %t.c
// RUN: %clang_cc1 -fsyntax-only -verify %t.c
// expected-no-diagnostics
// REQUIRES: shell

#include "rewrite-includes-bom.h"
