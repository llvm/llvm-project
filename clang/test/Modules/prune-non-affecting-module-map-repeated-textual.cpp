// Same as prune-non-affecting-module-map-repeated.cpp, but check that textual-only
// inclusions do not cause duplication of the module map files they are defined in.

// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file=%t/base.map -fmodule-map-file=%t/mixed.map\
// RUN:   -fmodule-map-file=%t/mod1.map \
// RUN:   -fmodule-name=mod1 -emit-module %t/mod1.map -o %t/mod1.pcm
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file=%t/mixed.map\
// RUN:   -fmodule-name=mixed -emit-module %t/mixed.map -o %t/mixed.pcm
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file=%t/base.map -fmodule-map-file=%t/mod1.map -fmodule-map-file=%t/mod2.map \
// RUN:   -fmodule-file=%t/mod1.pcm -fmodule-file=%t/mixed.pcm \
// RUN:   -fmodule-name=mod2 -emit-module %t/mod2.map -o %t/mod2.pcm
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file=%t/base.map -fmodule-map-file=%t/mod2.map -fmodule-map-file=%t/mod3.map \
// RUN:   -fmodule-file=%t/mod2.pcm -fmodule-file=%t/mixed.pcm \
// RUN:   -fmodule-name=mod3 -emit-module %t/mod3.map -o %t/mod3.pcm
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file=%t/base.map -fmodule-map-file=%t/mod3.map -fmodule-map-file=%t/mod4.map \
// RUN:   -fmodule-file=%t/mod3.pcm \
// RUN:   -fmodule-name=mod4 -emit-module %t/mod4.map -o %t/mod4.pcm
// RUN: %clang_cc1 -xc++ -fmodules -fno-implicit-modules -fmodule-map-file=%t/base.map -fmodule-map-file=%t/mod4.map -fmodule-file=%t/mod4.pcm -fsyntax-only -verify %t/check_slocs.cc

//--- base.map
module base { textual header "vector.h" }
//--- mixed.map
module mixed { textual header "mixed_text.h" header "mixed_mod.h"}
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
#include "vector.h"
#pragma clang __debug sloc_usage // expected-remark {{source manager location address space usage}}
// expected-note@* {{% of available space}}

// base.map must only be present once, despite being used in each module.
// Because its location in every module compile should be non-affecting.

// expected-note@base.map:1 {{file entered 1 time}}

// different modules use either only textual header from mixed.map or both textual and modular
// headers. Either combination must lead to only 1 use at the end, because the module is ultimately
// in the import chain and any textual uses should not change that.

// expected-note@mixed.map:1 {{file entered 1 time}}

// expected-note@* + {{file entered}}


//--- vector.h
#ifndef VECTOR_H
#define VECTOR_H
#endif

//--- mixed_text.h
#ifndef MIXED_TEXT_H
#define MIXED_TEXT_H
#endif
//--- mixed_mod.h
#ifndef MIXED_MOD_H
#define MIXED_MOD_H
#endif

//--- mod1.h
#ifndef MOD1
#define MOD1
#include "vector.h"
#include "mixed_text.h"
int mod1();
#endif
//--- mod2.h
#ifndef MOD2
#define MOD2
#include "vector.h"
#include "mod1.h"
#include "mixed_mod.h"
int mod2();
#endif
//--- mod3.h
#ifndef MOD3
#define MOD3
#include "vector.h"
#include "mod2.h"
#include "mixed_text.h"
#include "mixed_mod.h"
int mod3();
#endif
//--- mod4.h
#ifndef MOD4
#define MOD4
#include "vector.h"
#include "mod3.h"
int mod4();
#endif
