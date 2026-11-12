// RUN: rm -rf %t
// RUN: %clang_cc1 -include "g.h" -include "e.h" -include "f.h" -include "i.h" -fimplicit-module-maps -fmodules-cache-path=%t -fmodules-decluse -fmodule-name=XG -I %S/Inputs/declare-use %s -verify
// expected-error {{module XG does not directly depend on a module exporting 'f.h', which is part of indirectly-used module XF}}
const int g2 = g1 + e + f + aux_i;
