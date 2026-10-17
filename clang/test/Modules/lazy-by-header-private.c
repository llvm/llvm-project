// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t \
// RUN:   -fmodules-cache-path=%t/cache %t/tu.c -fsyntax-only -Rmodule-map \
// RUN:   -fmodules-lazy-load-module-maps -verify

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t \
// RUN:   -fmodules-cache-path=%t/cache %t/tu.c -fsyntax-only -Rmodule-map \
// RUN:   -DEAGER -verify

// Test that headers declared in private module maps are found correctly with
// lazy loading.

//--- module.modulemap

module A {
  header "A.h"
}

//--- module.private.modulemap

module A_Private {
  header "A_Private.h"
}

//--- A.h

//--- A_Private.h

//--- tu.c

#include <A_Private.h>

#ifndef EAGER
// expected-remark@*{{parsing modulemap}}
// expected-remark@*{{parsing modulemap}}
// expected-remark@*{{loading parsed module 'A_Private'}}
#else
// expected-remark@*{{loading modulemap}}
// expected-remark@*{{loading modulemap}}
#endif
