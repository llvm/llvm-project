// Check that the same module map file passed to -fmodule-map-file *and*
// available from one of the `-fmodule-file` does not allocate extra source
// location space. This optimization is important for using module maps in
// large codebases to avoid running out of source location space.

// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules -fmodule-map-file=%t/base.map -fmodule-name=base -emit-module %t/base.map -o %t/base.pcm
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file=%t/base.map -fmodule-map-file=%t/mod1.map \
// RUN:   -fmodule-file=%t/base.pcm \
// RUN:   -fmodule-name=mod1 -emit-module %t/mod1.map -o %t/mod1.pcm
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file=%t/base.map -fmodule-map-file=%t/mod1.map -fmodule-map-file=%t/mod2.map \
// RUN:   -fmodule-file=%t/mod1.pcm \
// RUN:   -fmodule-name=mod2 -emit-module %t/mod2.map -o %t/mod2.pcm
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file=%t/base.map -fmodule-map-file=%t/mod2.map -fmodule-map-file=%t/mod3.map \
// RUN:   -fmodule-file=%t/mod2.pcm \
// RUN:   -fmodule-name=mod3 -emit-module %t/mod3.map -o %t/mod3.pcm
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file=%t/base.map -fmodule-map-file=%t/mod3.map -fmodule-map-file=%t/mod4.map \
// RUN:   -fmodule-file=%t/mod3.pcm \
// RUN:   -fmodule-name=mod4 -emit-module %t/mod4.map -o %t/mod4.pcm
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules -fmodule-map-file=%t/base.map -fmodule-map-file=%t/mod4.map -fmodule-file=%t/mod4.pcm -fsyntax-only -verify %t/check_slocs.cc

//--- base.map
module base { header "vector.h" }
//--- mod1.map
module mod1 { header "mod1.h" }
//--- mod2.map
module mod2 { header "mod2.h" }
//--- mod3.map
module mod3 { header "mod3.h" }
//--- mod4.map
module mod4 { header "mod4.h" }
//--- check_slocs.cc
#include "mod4.h"
#pragma clang __debug sloc_usage // expected-remark {{source manager location address space usage}}
// expected-note@* {{% of available space}}

// Module map files files that were specified on the command line are entered twice (once when parsing command-line, once loaded from the .pcm)
// Those that  not specified on the command line must be entered once.

// expected-note@base.map:1 {{file entered 2 times}}
// expected-note@mod4.map:1 {{file entered 2 times}}
// expected-note@mod1.map:1 {{file entered 1 time}}
// expected-note@mod2.map:1 {{file entered 1 time}}
// expected-note@mod3.map:1 {{file entered 1 time}}
// expected-note@* + {{file entered}}


//--- vector.h
#ifndef VECTOR_H
#define VECTOR_H
#endif

//--- mod1.h
#ifndef MOD1
#define MOD1
#include "vector.h"
int mod1();
#endif
//--- mod2.h
#ifndef MOD2
#define MOD2
#include "vector.h"
#include "mod1.h"
int mod2();
#endif
//--- mod3.h
#ifndef MOD3
#define MOD3
#include "vector.h"
#include "mod2.h"
int mod3();
#endif
//--- mod4.h
#ifndef MOD4
#define MOD4
#include "vector.h"
#include "mod3.h"
int mod4();
#endif
