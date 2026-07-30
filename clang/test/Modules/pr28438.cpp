// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -verify %s -fmodules -fmodules-cache-path=%t -I%S/Inputs/PR28438 -fimplicit-module-maps

#include "a.h"
#include "b2.h"

FOO // expected-no-diagnostics
