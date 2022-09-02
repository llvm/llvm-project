// RUN: rm -rf %t
// RUN: %clang_cc1 -fimplicit-module-maps -fmodules-decluse -fmodule-name=XG -I %S/Inputs/declare-use %s -verify
//
// Check these flags get properly passed through the driver even when -fmodules is disabled.
// RUN: %clang -fimplicit-module-maps -fmodules-decluse -fmodule-name=XG -I %S/Inputs/declare-use -c -o /dev/null -Xclang -verify %s

#include "g.h"
#include "e.h"
#include "f.h" // expected-error {{module XG does not depend on a module exporting 'f.h'}}
#include "i.h"
#include "sub.h"
const int g2 = g1 + e + f + aux_i + sub;
