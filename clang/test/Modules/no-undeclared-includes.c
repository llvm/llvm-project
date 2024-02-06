// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %t %t/no-undeclared-includes.c -verify

//--- no-undeclared-includes.c
// expected-no-diagnostics
#include <assert.h>

//--- assert.h
#include <base.h>

//--- base.h
#ifndef base_h
#define base_h



#endif /* base_h */

//--- module.modulemap
module cstd [system] [no_undeclared_includes] {
  use base
  module assert {
    textual header "assert.h"
  }
}

module base [system] {
  header "base.h"
  export *
}
