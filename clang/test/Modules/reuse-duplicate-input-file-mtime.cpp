// Check that reusing an input file of a loaded module follows the modification
// time the module recorded, the way the header file info table does.

// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

// Only an implicitly built module records a modification time, so mod1 is
// built that way to get a nonzero one to compare against.
// RUN: %clang_cc1 -xc++ -fmodules -fimplicit-module-maps -fdisable-module-hash \
// RUN:   -fmodules-cache-path=%t/cache -I%t -fsyntax-only %t/tu1.cc

// The file still has the time mod1 recorded, so mod2 reuses mod1's copy.
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file=%t/module.modulemap -fmodule-file=%t/cache/mod1.pcm \
// RUN:   -I%t -fmodule-name=mod2 -emit-module %t/module.modulemap -o %t/match.pcm
// RUN: llvm-bcanalyzer -dump %t/match.pcm | FileCheck %s --check-prefix=REUSED

// Giving the file a different time leaves mod1's copy unusable, so mod2 writes
// its own entries. Validation is off so that mod1 is kept rather than rebuilt.
// RUN: touch -m -a -t 202901010000 %t/shared.h
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file=%t/module.modulemap -fmodule-file=%t/cache/mod1.pcm \
// RUN:   -I%t -fmodule-name=mod2 -fno-validate-pch \
// RUN:   -emit-module %t/module.modulemap -o %t/diverged.pcm
// RUN: llvm-bcanalyzer -dump %t/diverged.pcm | FileCheck %s --check-prefix=OWN

// An explicitly built mod1 records no time at all, so there is nothing to
// disagree with and mod2 reuses its copy however the file is stamped.
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file=%t/module.modulemap -I%t -fmodule-name=mod1 \
// RUN:   -emit-module %t/module.modulemap -o %t/mod1-explicit.pcm
// RUN: touch -m -a -t 203001010000 %t/shared.h
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file=%t/module.modulemap \
// RUN:   -fmodule-file=%t/mod1-explicit.pcm -I%t -fmodule-name=mod2 \
// RUN:   -fno-validate-pch -emit-module %t/module.modulemap -o %t/untimed.pcm
// RUN: llvm-bcanalyzer -dump %t/untimed.pcm | FileCheck %s --check-prefix=REUSED

// The input file record keeps the index of the first source location entry
// written for the file, which is zero when the module reuses a loaded copy.

// REUSED: <INPUT_FILE {{.*}}op7=0 op8=0{{.*}}shared.h
// OWN: <INPUT_FILE {{.*}}op7={{[1-9][0-9]*}} {{.*}}shared.h

//--- module.modulemap
module mod1 { header "mod1.h" export * }
module mod2 { header "mod2.h" export * }

//--- shared.h
#ifndef SHARED_H
#define SHARED_H
struct Shared { int a; };
int shared_fn(void);
#endif

//--- mod1.h
#include "shared.h"
int mod1_fn(void);

//--- mod2.h
#include "shared.h"
int mod2_fn(void);

//--- tu1.cc
#include "mod1.h"
