// REQUIRES: hexagon-registered-target

// Verify that hexagon_types.h can be included after a B0 macro definition
// without conflicts, that B0 is restored afterward, and that the
// lowercase b0() alias is usable even while B0 is still a macro.

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

// In C++ mode, verify the lowercase b0() alias works even with B0 defined.
#ifdef __cplusplus
void test_b0_alias(void) {
  HEXAGON_Vect64C v(0x0807060504030201LL);
  signed char got = v.b0();
  (void)got;
  HEXAGON_Vect64C v2 = v.b0(0x42);
  (void)v2;
}
#endif
