// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/merge-anon-in-template -verify %s
// expected-no-diagnostics
#include "a.h"
#include "c.h"
is_floating<int>::type t;
