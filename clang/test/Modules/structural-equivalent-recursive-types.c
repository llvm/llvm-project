// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/rec-types -fsyntax-only %s -verify
#include "a.h"
#include "c.h"
void foo(struct some_descriptor *st) { (void)st->thunk; }

// expected-no-diagnostics
