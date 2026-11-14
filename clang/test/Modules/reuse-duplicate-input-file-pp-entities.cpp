// Check that reusing a duplicate input file keeps a module file's preprocessed
// entities in the order a reader looks them up in. The entities of a reused
// file are redirected into the module that kept it, and a reader orders a
// location it loaded before any of this module file's own, so they have to be
// written before the rest or the ones after them cannot be found.

// RUN: rm -rf %t && split-file %s %t

// mod2 reuses mod1's copy of shared.h, so the entities shared.h contributed to
// mod2 name locations in mod1. M2_POST is defined after that, and is the one
// that goes missing when the order is wrong.
// RUN: c-index-test -cursor-at=%t/mod2.h:6:9 %t/tu.cpp -fmodules \
// RUN:   -fimplicit-module-maps -I%t -fmodules-cache-path=%t/mc-reuse \
// RUN:   -Xclang -fdisable-module-hash | FileCheck %s

// With -DNO_MOD1, mod2 doesn't load mod1 and reuses nothing.
// RUN: c-index-test -cursor-at=%t/mod2.h:6:9 %t/tu.cpp -fmodules \
// RUN:   -fimplicit-module-maps -I%t -fmodules-cache-path=%t/mc-control \
// RUN:   -DNO_MOD1 -Xclang -fdisable-module-hash | FileCheck %s

// CHECK: 6:9 macro definition=M2_POST

//--- module.modulemap
module mod1 { header "mod1.h" export * }
module mod2 { header "mod2.h" export * }

//--- inner.h
#ifndef INNER_H
#define INNER_H
#define INNER_A 1
int inner_fn(void);
#endif

//--- shared.h
#ifndef SHARED_H
#define SHARED_H
#define SH_ONE 1
#include "inner.h"
#define SH_TWO (SH_ONE + INNER_A)
int shared_fn(void);
#endif

//--- mod1.h
#define M1_PRE 0
#include "shared.h"

//--- mod2.h
#define M2_PRE 0
#include "shared.h"
#ifndef NO_MOD1
#include "mod1.h"
#endif
#define M2_POST (SH_TWO)

//--- tu.cpp
#include "mod2.h"
int u(void) { return SH_TWO + M2_POST; }
