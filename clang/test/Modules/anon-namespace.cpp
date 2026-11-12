// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/anon-namespace -verify %s
// expected-no-diagnostics
#include "b1.h"
#include "c.h"
using namespace N;
