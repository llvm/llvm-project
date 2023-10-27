// RUN: rm -rf %t
// RUN: %clang_cc1 -fimplicit-module-maps -fmodules-cache-path=%t -fmodules-strict-decluse -fmodule-name=XG -I %S/Inputs/declare-use %s -verify
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fmodules-strict-decluse -fmodule-name=XG -I %S/Inputs/declare-use %s -verify -fno-modules-error-recovery

#define GIMME_A_K
#include "k.h"

#define GIMME_AN_L
#include "l.h" // expected-error {{module XG does not directly depend on a module exporting 'l.h', which is part of indirectly-used module XL}}

#include "m2.h" // expected-error {{module XG does not directly depend on a module exporting 'm2.h', which is part of indirectly-used module XM}}
const int use_m = m; // expected-error {{undeclared identifier}}

#define GIMME_AN_M
#include "m.h" // expected-error {{use of private header from outside its module: 'm.h'}}
const int use_m_2 = m;

const int g = k + l;
