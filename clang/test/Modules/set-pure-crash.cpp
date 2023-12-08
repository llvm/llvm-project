// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x c++ -I %S/Inputs/set-pure-crash -verify %s -o %t

// expected-no-diagnostics

#include "b.h"
#include "c.h"

auto t = simple<const char *>();
