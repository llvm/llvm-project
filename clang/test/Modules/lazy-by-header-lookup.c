// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t \
// RUN:   -fmodules-cache-path=%t/cache %t/tu.c -fsyntax-only -Rmodule-map \
// RUN:   -fmodules-lazy-load-module-maps -verify

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t \
// RUN:   -fmodules-cache-path=%t/cache %t/tu.c -fsyntax-only -Rmodule-map \
// RUN:   -DEAGER -verify

//--- module.modulemap

module A {
  header "A.h"
}

module B {
  header "B.h"
  header "B2.h"
}

//--- A.h

//--- B.h

//--- B2.h

//--- tu.c

#include <B.h>
#include <B2.h>

#ifndef EAGER
// expected-remark@*{{parsing modulemap}}
// expected-remark@*{{loading parsed module 'B'}}
#else
// expected-remark@*{{loading modulemap}}
#endif
