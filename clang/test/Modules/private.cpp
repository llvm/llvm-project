// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/private %s -verify
// RUN: %clang_cc1 -x objective-c -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/private %s -fsyntax-only -Wno-private-header

#include "common.h"
@import libPrivate1;
#include "private1.h" // expected-error {{use of private header from outside its module}}
#include "public2.h"
#include "private2.h" // expected-error {{use of private header from outside its module}}

struct use_this1 client_variable1;
struct use_this2 client_variable2;
struct mitts_off1 client_variable3;
struct mitts_off2 client_variable4;
