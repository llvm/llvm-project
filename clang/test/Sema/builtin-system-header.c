// RUN: %clang_cc1 -fms-extensions -fsyntax-only -verify -triple arm64-windows -isystem %S/Inputs %s

// expected-no-diagnostics
#include <builtin-system-header.h>

void foo() {
  MACRO(0,0);
}
