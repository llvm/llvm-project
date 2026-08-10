// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fmodules-local-submodule-visibility -I%S/Inputs/internal-constants %s -verify

// expected-no-diagnostics
#include "c.h"

int q = h();
int r = N::k;

#include "b.h"

int s = N::k; // FIXME: This should be ambiguous if we really want internal linkage declarations to not collide.
