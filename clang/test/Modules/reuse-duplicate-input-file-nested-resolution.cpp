// Check that reusing a duplicate input file does not carry over the files it
// included. The same shared.h is read by both modules, but each resolves the
// inner.h it includes against a different search path, so the two copies of
// the subtree below shared.h stand for different files and only shared.h
// itself may be reused.

// RUN: rm -rf %t && split-file %s %t

// mod1 resolves inner.h to one/inner.h, mod2 to two/inner.h. The two are the
// same size, so nothing about shared.h or the extent it spans tells them apart.
// RUN: %clang_cc1 -x c++ -fmodules -fimplicit-module-maps -fdisable-module-hash \
// RUN:   -I%t -I%t/one -emit-module -fmodule-name=mod1 %t/module.modulemap \
// RUN:   -o %t/mod1.pcm
// RUN: %clang_cc1 -x c++ -fmodules -fimplicit-module-maps -fdisable-module-hash \
// RUN:   -I%t -I%t/two -fmodule-file=%t/mod1.pcm -emit-module -fmodule-name=mod2 \
// RUN:   %t/module.modulemap -o %t/mod2.pcm

// mod2 reuses mod1's copy of shared.h, so check that bbb_fn is still reported
// in the file mod2 read it from and not in the one mod1 read.
// RUN: %clang_cc1 -x c++ -fmodules -fimplicit-module-maps -fdisable-module-hash \
// RUN:   -I%t -I%t/two -fmodule-file=%t/mod2.pcm -fsyntax-only %t/tu.cpp 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK: warning: 'bbb_fn' is deprecated
// CHECK-NEXT: int use(void) { return bbb_fn(); }
// CHECK: [[PREFIX]]/two/inner.h:1:{{[0-9]+}}: note: 'bbb_fn' has been explicitly marked deprecated here
// CHECK-NEXT: int bbb_fn(void) __attribute__((deprecated));

//--- module.modulemap
module mod1 { header "mod1.h" export * }
module mod2 { header "mod2.h" export * }

//--- one/inner.h
int aaa_fn(void) __attribute__((deprecated));

//--- two/inner.h
int bbb_fn(void) __attribute__((deprecated));

//--- shared.h
#ifndef SHARED_H
#define SHARED_H
#include "inner.h"
int shared_fn(void);
#endif

//--- mod1.h
#include "shared.h"

//--- mod2.h
#include "shared.h"
#include "mod1.h"

//--- tu.cpp
#include "mod2.h"
int use(void) { return bbb_fn(); }
