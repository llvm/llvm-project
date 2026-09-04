// Check that multiple include-once of a header not covered by an umbrella work.

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -fsyntax-only %t/tu.c -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -verify 

//--- module.modulemap
module M {
  umbrella header "M.h"
  module * { export * }
}
//--- M.h
//--- NotCovered.h
//--- tu.c
#import "NotCovered.h" // expected-warning{{missing submodule 'M.NotCovered'}}
#import "NotCovered.h" // expected-warning{{missing submodule 'M.NotCovered'}}
