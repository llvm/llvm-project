// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/inc/module.modulemap.in > %t/inc/module.modulemap
// RUN: %clang_cc1 -I%t/inc -I%t/sys -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -fsyntax-only -verify %t/main.c

// A module map may list a header by absolute path. Looking that path up does
// not go through a search directory, which must not crash '#include_next'.

//--- inc/module.modulemap.in
module Wrapper {
  header "DIR/inc/wrapper.h"
  export *
}

//--- inc/wrapper.h
#ifndef WRAPPER_H
#define WRAPPER_H
#include_next <target.h>
#endif

//--- inc/target.h
#error "target.h in inc must not be picked"

//--- sys/target.h
int target(void);

//--- main.c
// expected-no-diagnostics
#include <wrapper.h>
int use(void) { return target(); }
