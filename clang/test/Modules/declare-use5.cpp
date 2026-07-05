// RUN: rm -rf %t
// RUN: %clang_cc1 -fimplicit-module-maps -fmodules-cache-path=%t -fmodules-decluse -fmodule-name=XN -I %S/Inputs/declare-use %s -verify


#include "sub.h"

const int a = sub;

// expected-no-diagnostics
