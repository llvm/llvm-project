// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

#include <stddef.h>

typedef void (*hookfunc)(void *arg);
hookfunc hook;

void clear_hook() {
  hook = NULL;
}
