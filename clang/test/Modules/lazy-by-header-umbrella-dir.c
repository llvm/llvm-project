// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t \
// RUN:   -fmodules-cache-path=%t/cache %t/tu.c -fsyntax-only -Rmodule-map \
// RUN:   -fmodules-lazy-load-module-maps -verify

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t \
// RUN:   -fmodules-cache-path=%t/cache %t/tu.c -fsyntax-only -Rmodule-map \
// RUN:   -DEAGER -verify

// Test that umbrella directories trigger module loading. When a header is
// included that falls under an umbrella directory, the module declaring that
// umbrella directory should be loaded. Other modules in the same module map
// should not be loaded unless their headers are also included.

//--- module.modulemap

module A {
  umbrella "subdir"
}

// These modules shouldn't get loaded.
module Other {
  umbrella "othersubdir"
}

module B {
  header "B.h"
}

//--- subdir/foo.h

//--- subdir/bar.h

//--- othersubdir/os.h

//--- B.h

//--- C.h

// Test umbrella "." which covers all headers in the module map's directory.
//--- dotumbrella/module.modulemap

module DotUmbrella {
  umbrella "."
}

//--- dotumbrella/dot.h

//--- tu.c

// Including a header from the umbrella directory should load module A.
#include <subdir/foo.h>
#include <subdir/bar.h>
#include <C.h> // This shouldn't trigger any loads.
#include <dotumbrella/dot.h>

#ifndef EAGER
// expected-remark@*{{parsing modulemap}}
// expected-remark@*{{loading parsed module 'A'}}
// expected-remark@*{{parsing modulemap}}
// expected-remark@*{{loading parsed module 'DotUmbrella'}}
#else
// expected-remark@*{{loading modulemap}}
// expected-remark@*{{loading modulemap}}
#endif
