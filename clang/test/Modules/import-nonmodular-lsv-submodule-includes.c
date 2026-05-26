// Test that #import correctly skips a non-modular header under local submodule
// visibility when the header is included by a submodule (not a top-level module).

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- include/Textual.h
struct my_struct {
  int field;
};

//--- include/A/module.modulemap
module A {
  module A1 { header "A1.h" export * }
  module A2 { header "A2.h" }
  export *
}

//--- include/A/A1.h
#include "Textual.h"

//--- include/A/A2.h

//--- include/B/module.modulemap
module B {
  module B1 { header "B1.h" export * }
  module B2 { header "B2.h" export * }
}

//--- include/B/B1.h
#import "A/A1.h"
#import "Textual.h"
static struct my_struct x;

//--- include/B/B2.h
#import "A/A1.h"
#import "Textual.h"
static struct my_struct y;

//--- tu.c
#import "B/B1.h"
#import "B/B2.h"

// Build module A.
// RUN: %clang_cc1 -fmodules -I %t/include -emit-module %t/include/A/module.modulemap -fmodule-name=A -o %t/A.pcm

// Build module B with local submodule visibility.
// RUN: %clang_cc1 -fmodules -fmodules-local-submodule-visibility -I %t/include -emit-module %t/include/B/module.modulemap -fmodule-name=B -fmodule-file=%t/A.pcm -o %t/B.pcm

// Use module B.
// RUN: %clang_cc1 -fmodules -fmodules-local-submodule-visibility -I %t/include -fsyntax-only %t/tu.c -fmodule-file=%t/B.pcm -verify

// expected-no-diagnostics
