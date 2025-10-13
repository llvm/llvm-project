// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t \
// RUN:   -fmodules-cache-path=%t/cache %t/tu.c -fsyntax-only -Rmodule-map \
// RUN:   -verify

//--- module.modulemap

module A {
  header "A.h"
}

module B {
  header "B.h"
}

//--- A.h

//--- B.h

//--- tu.c

#pragma clang __debug module_lookup A // does module map search for A
#pragma clang __debug module_map A // A is now in the ModuleMap,
#pragma clang __debug module_map B // expected-warning{{unknown module 'B'}}
                                   // but B isn't.
#include <B.h> // Now load B via header search

// expected-remark@*{{parsing modulemap}}
// expected-remark@*{{loading parsed module 'A'}}
// expected-remark@*{{loading modulemap}}