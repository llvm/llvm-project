// RUN: rm -rf %t
// RUN: %clang_cc1 -fimplicit-module-maps -fmodules-cache-path=%t -fmodules-decluse -fmodule-name=XH -I %S/Inputs/declare-use %s -verify

#include "h.h"
#include "e.h"
#include "f.h" // expected-error {{module XH does not directly depend on a module exporting 'f.h', which is part of indirectly-used module XF}}
const int h2 = h1+e+f;
