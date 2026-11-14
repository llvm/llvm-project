// Check that reusing an input file of a loaded module keeps the import stack
// intact. A module's import location is written local to the module file, so
// the file it points into has to keep its own source location entries.
//
// This asserts diagnostic text on purpose: the import stack is what the fix
// restores. If the stack's wording changes upstream, update these lines rather
// than dropping them.

// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

// The shared header is entered by mod1 and by mod2, and imports mod0. mod2
// reuses mod1's copy of it, so mod0's import location points into a file mod2
// did not write entries for.
// RUN: %clang_cc1 -x c++ -std=c++17 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/plain/mc -I%t/plain -fsyntax-only \
// RUN:   %t/plain/tu.cpp 2>&1 | sed 's:\\\\\?:/:g' | FileCheck %s --check-prefix=PLAIN

// PLAIN:      In module 'mod2' imported from
// PLAIN-NEXT: In module 'mod0' imported from {{.*}}/shared.h:4:
// PLAIN-NEXT: {{.*}}/mod0.h:1:{{.*}}warning: 'old' is deprecated

// The same shape, except a macro decides whether the shared header imports
// mod0, so the two copies of it do not agree on what they import.
// RUN: %clang_cc1 -x c++ -std=c++17 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/macro/mc -I%t/macro -fsyntax-only \
// RUN:   %t/macro/tu.cpp 2>&1 | sed 's:\\\\\?:/:g' | FileCheck %s --check-prefix=MACRO

// MACRO:      In module 'mod2' imported from
// MACRO-NEXT: In module 'mod0' imported from {{.*}}/shared.h:3:
// MACRO-NEXT: {{.*}}/mod0.h:1:{{.*}}warning: 'old' is deprecated

//--- plain/module.modulemap
module mod0 { header "mod0.h" export * }
module mod1 { header "mod1.h" export * }
module mod2 { header "mod2.h" export * }

//--- plain/shared.h
#ifndef SHARED_H
#define SHARED_H
int shared_fn(void);
#include "mod0.h"
#endif

//--- plain/mod0.h
template <class T> int mod0_t(T t) { return t.old(); }

//--- plain/mod1.h
#include "shared.h"
int mod1_fn(void);

//--- plain/mod2.h
#include "shared.h"
#include "mod1.h"
int mod2_fn(void);

//--- plain/tu.cpp
#include "mod2.h"
struct S { [[deprecated]] int old(); };
int use = mod0_t(S());

//--- macro/module.modulemap
module mod0 { header "mod0.h" export * }
module mod1 { header "mod1.h" export * }
module mod2 { header "mod2.h" export * }

//--- macro/shared.h
int shared_fn(void);
#ifndef NO_MOD0
#include "mod0.h"
#endif

//--- macro/mod0.h
template <class T> int mod0_t(T t) { return t.old(); }

//--- macro/mod1.h
#define NO_MOD0 1
#include "shared.h"
#undef NO_MOD0
int mod1_fn(void);

//--- macro/mod2.h
#include "mod1.h"
#include "shared.h"
int mod2_fn(void);

//--- macro/tu.cpp
#include "mod2.h"
struct S { [[deprecated]] int old(); };
int use = mod0_t(S());
