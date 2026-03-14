// RUN: rm -rf %t
// RUN: %clang_cc1 -fimplicit-module-maps -fmodules-cache-path=%t -fmodules-decluse -fmodule-name=XG -I %S/Inputs/declare-use %s -verify

#define ALLOWED_INC "b.h"

#include "j.h"

const int g2 = j;

// expected-no-diagnostics
