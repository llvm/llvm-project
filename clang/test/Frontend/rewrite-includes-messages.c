// RUN: %clang_cc1 -E -frewrite-includes %s -I%S/Inputs/ | %clang_cc1 -Wall -fsyntax-only -Wunused-macros -x c - 2>&1 > %t.1
// RUN: %clang_cc1 -I%S/Inputs/ -Wall -Wunused-macros -fsyntax-only %s 2>&1 > %t.2
// RUN: diff %t.1 %t.2 -u
// expected-no-diagnostics

#include "rewrite-includes-messages.h"
#define UNUSED_MACRO
