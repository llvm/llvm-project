// Check that a macro expanded inside a reused header stays comparable with the
// rest of this module file. The expansion entry stays here while the file it
// expanded in is left out, so its location names the module that kept the
// header and the two are ordered in the source location address space.

// RUN: rm -rf %t && split-file %s %t

// Both modules declare f(int *) through a macro expanded in shared.h. mod2
// reuses mod1's copy of shared.h, but keeps its own entry for the expansion.
// RUN: not %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/mc-reuse -I%t %t/tu.cpp 2>&1 | FileCheck %s

// With -DNO_MOD1, mod2 doesn't load mod1 and reuses nothing.
// RUN: not %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/mc-control -I%t -DNO_MOD1 %t/tu.cpp 2>&1 \
// RUN:   | FileCheck %s

// CHECK: error: no matching function for call to 'f'
// CHECK-DAG: shared.h:5:6: note: candidate function not viable
// CHECK-DAG: mod2.h:5:6: note: candidate function not viable

//--- module.modulemap
module mod1 { header "mod1.h" export * }
module mod2 { header "mod2.h" export * }

//--- shared.h
#ifndef SHARED_H
#define SHARED_H
#define DECL(name, type) void name(type);
int shared_fn(void);
DECL(f, int *)
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
