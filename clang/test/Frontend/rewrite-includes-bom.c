// RUN: %clang -E -frewrite-includes -I %S/Inputs %s -o - | %clang -fsyntax-only -Xclang -verify -x c -
// expected-no-diagnostics

#include "rewrite-includes-bom.h"
