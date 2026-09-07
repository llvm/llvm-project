// Check that a file two modules both include textually only takes up source
// location space once. A module that already has the file lends its copy to
// the next one, which points its own locations at that copy instead of
// writing a second set of entries for the same text.
//
// A file that something still names by FileID has to keep its own entries,
// since a FileID only means anything in the module file that wrote it. Those
// are the files with a line table entry or with diagnostic state of their own.

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
// RUN:   -fsyntax-only -verify %t/check_slocs.cc

//--- mods.map
module mod1 { header "mod1.h" export * }
module mod2 { header "mod2.h" export * }

//--- check_slocs.cc
#include "mod2.h"
#pragma clang __debug sloc_usage // expected-remark {{source manager location address space usage}}
// expected-note@* {{% of available space}}

// Both modules include this textually, and nothing names it by FileID, so
// mod2 points at mod1's copy and the file is entered once.

// expected-note@shared.h:1 {{file entered 1 time}}

// Both modules include these textually as well, but each is still named by
// FileID somewhere, so each module keeps its own entries for them.

// expected-note@lines.h:1 {{file entered 2 times}}
// expected-note@diags.h:1 {{file entered 2 times}}

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
// A line directive puts this file in the line table, which the module file
// records by FileID.
#line 500 "somewhere-else.h"
#endif

//--- diags.h
#ifndef DIAGS_H
#define DIAGS_H
int diags_fn(void);
// A diagnostic pragma gives this file diagnostic state of its own, which the
// module file also records by FileID.
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
