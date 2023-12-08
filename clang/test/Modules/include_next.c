// RUN: rm -rf %t
// RUN: %clang_cc1 -I%S/Inputs/include_next/x -I%S/Inputs/include_next/y -verify %s
// RUN: %clang_cc1 -I%S/Inputs/include_next/x -I%S/Inputs/include_next/y -verify %s -fmodules -fimplicit-module-maps -fmodules-cache-path=%t

// expected-no-diagnostics
#include "a.h"
#include "subdir/b.h"
_Static_assert(ax == 1, "");
_Static_assert(ay == 2, "");
_Static_assert(bx == 3, "");
_Static_assert(by == 4, "");
