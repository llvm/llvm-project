// Check that a header included textually by several modules reuses source
// location entries, while headers named by FileID do not.

// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file=%t/mods.map \
// RUN:   -fmodule-name=mod1 -emit-module %t/mods.map -o %t/mod1.pcm
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file=%t/mods.map -fmodule-file=%t/mod1.pcm \
// RUN:   -fmodule-name=mod2 -emit-module %t/mods.map -o %t/mod2.pcm
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file=%t/mods.map -fmodule-file=%t/mod2.pcm \
// RUN:   -fmodule-name=mod3 -emit-module %t/mods.map -o %t/mod3.pcm
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file=%t/mods.map -fmodule-file=%t/mod3.pcm \
// RUN:   -fsyntax-only -verify %t/check_slocs.cc

// The modules are chained through -fmodule-file rather than including one
// another, which would carry include guards along and avoid textual entry.

//--- mods.map
module mod1 { header "mod1.h" export * }
module mod2 { header "mod2.h" export * }
module mod3 { header "mod3.h" export * }

//--- check_slocs.cc
#include "mod3.h"
#pragma clang __debug sloc_usage // expected-remark {{source manager location address space usage}}
// expected-note@* {{% of available space}}

// shared.h is entered once. mod2 redirects to mod1's copy, and mod3 finds
// that copy through mod2.

// expected-note@shared.h:1 {{file entered 1 time}}

// lines.h and diags.h are named by FileID through the line table and
// diagnostic state, so they cannot be redirected.

// expected-note@lines.h:1 {{file entered 3 times}}
// expected-note@diags.h:1 {{file entered 3 times}}

// expected-note@* + {{file entered}}

//--- shared.h
#ifndef SHARED_H
#define SHARED_H
struct Shared {
  int a;
};
int shared_fn(void);
#endif

//--- lines.h
#ifndef LINES_H
#define LINES_H
int lines_fn(void);
#line 500 "somewhere-else.h"
#endif

//--- diags.h
#ifndef DIAGS_H
#define DIAGS_H
int diags_fn(void);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic pop
#endif

//--- mod1.h
#include "shared.h"
#include "lines.h"
#include "diags.h"
int mod1_fn(void);

//--- mod2.h
#include "shared.h"
#include "lines.h"
#include "diags.h"
int mod2_fn(void);

//--- mod3.h
#include "shared.h"
#include "lines.h"
#include "diags.h"
int mod3_fn(void);
