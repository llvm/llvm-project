// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t \
// RUN:   -fmodules-cache-path=%t/cache %t/tu.c -fsyntax-only -Rmodule-map \
// RUN:   -fmodules-lazy-load-module-maps -verify

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t \
// RUN:   -fmodules-cache-path=%t/cache %t/tu.c -fsyntax-only -Rmodule-map \
// RUN:   -DEAGER -verify

// Test that umbrella headers trigger module loading. Since the lazy loader
// can't know which headers are covered by an umbrella header without parsing
// it, modules with umbrella headers are loaded conservatively when any header
// in the same directory is included.

//--- module.modulemap

module A {
  umbrella header "A.h"
}

module B {
  header "B.h"
}

//--- A.h
#include <A_impl.h>

//--- A_impl.h
// A header that would be covered by the umbrella.

//--- B.h
// Not covered 

//--- tu.c

// Including a header that might be covered by an umbrella header should
// conservatively load the module with the umbrella header.
#include <A_impl.h>

#ifndef EAGER
// expected-remark@*{{parsing modulemap}}
// expected-remark@*{{loading parsed module 'A'}}
#else
// expected-remark@*{{loading modulemap}}
#endif
