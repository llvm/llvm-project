// Check that a file included from a reused header stays comparable with the
// rest of this module file. The file that included it is left out, so the two
// no longer meet at a common ancestor and are ordered by where they sit in the
// source location address space instead.

// RUN: rm -rf %t && split-file %s %t

// inner.h imports mod0, and shared.h includes it in both mod1 and mod2. mod2
// imports mod0 from inner.h, then loads mod1, so mod2 reuses mod1's copy of
// shared.h but keeps its own inner.h for the import location.
// RUN: not %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/mc-reuse -I%t %t/tu.cpp 2>&1 | FileCheck %s

// With -DNO_MOD1, mod2 doesn't load mod1 and reuses nothing.
// RUN: not %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/mc-control -I%t -DNO_MOD1 %t/tu.cpp 2>&1 \
// RUN:   | FileCheck %s

// CHECK: error: no matching function for call to 'f'
// CHECK-DAG: inner.h:4:6: note: candidate function not viable
// CHECK-DAG: mod2.h:5:6: note: candidate function not viable

//--- module.modulemap
module mod0 { header "mod0.h" export * }
module mod1 { header "mod1.h" export * }
module mod2 { header "mod2.h" export * }

//--- mod0.h
int mod0_fn(void);

//--- inner.h
#ifndef INNER_H
#define INNER_H
#include "mod0.h"
void f(int *);
#endif

//--- shared.h
#ifndef SHARED_H
#define SHARED_H
int shared_fn(void);
#include "inner.h"
#endif

//--- mod1.h
#include "shared.h"

//--- mod2.h
#include "shared.h"
#ifndef NO_MOD1
#include "mod1.h"
#endif
void f(double *);

//--- tu.cpp
#include "mod2.h"
void g() { f("s"); }
