// REQUIRES: hexagon-registered-target

// Verify that hexagon_types.h can be included after termios.h without
// B0 macro conflicts, and that B0 is restored afterward.

// RUN: %clang_cc1 -fsyntax-only -internal-isystem %S/../../lib/Headers/ \
// RUN:   -target-cpu hexagonv68 -triple hexagon-unknown-linux-musl \
// RUN:   -verify %s

// RUN: %clang_cc1 -fsyntax-only -internal-isystem %S/../../lib/Headers/ \
// RUN:   -target-cpu hexagonv68 -triple hexagon-unknown-linux-musl \
// RUN:   -x c++ -verify %s

// expected-no-diagnostics

// Simulate the POSIX termios.h B0 macro definition.
#define B0 0000000

#include <hexagon_types.h>

// Verify B0 is restored after including hexagon_types.h.
#ifndef B0
#error "B0 should be defined after including hexagon_types.h"
#endif

_Static_assert(B0 == 0, "B0 should still be 0 after including hexagon_types.h");
