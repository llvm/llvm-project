// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t \
// RUN:   -fmodules-cache-path=%t/cache %t/tu.c -fsyntax-only -Rmodule-map \
// RUN:   -fmodules-lazy-load-module-maps -verify

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t \
// RUN:   -fmodules-cache-path=%t/cache %t/tu.c -fsyntax-only -Rmodule-map \
// RUN:   -DEAGER -verify

// Test that module maps in parent directories are found when including headers
// from subdirectories. The module map declares headers with relative paths
// that include the subdirectory.

//--- module.modulemap

module A {
  header "sub/A.h"
}

module B {
  header "sub/deep/B.h"
}

module Other {
  header "other/O.h"
}

//--- sub/A.h

//--- sub/deep/B.h

//--- other/O.h

//--- tu.c

#include <sub/A.h>
#include <sub/deep/B.h>

#ifndef EAGER
// expected-remark@*{{parsing modulemap}}
// expected-remark@*{{loading parsed module 'A'}}
// expected-remark@*{{loading parsed module 'B'}}
#else
// expected-remark@*{{loading modulemap}}
#endif
