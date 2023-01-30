// RUN: rm -rf %t
// RUN: %clang_cc1 -fimplicit-module-maps -fmodules-cache-path=%t -fmodules-strict-decluse -fmodule-name=XG -I %S/Inputs/declare-use %s -verify

#include "g.h"
#include "e.h"
#include "f.h" // expected-error {{module XG does not directly depend on a module exporting 'f.h', which is part of indirectly-used module XF}}
#include "i.h" // expected-error {{module XG does not depend on a module exporting 'i.h'}}

const int g2 = g1 + e + f + aux_i;
