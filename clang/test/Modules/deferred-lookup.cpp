// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/deferred-lookup -verify %s
// expected-no-diagnostics

namespace N { int f(int); }
#include "b.h"
