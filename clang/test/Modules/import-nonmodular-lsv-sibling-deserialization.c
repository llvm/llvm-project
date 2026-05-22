// Test that #import correctly skips a non-modular header under local submodule
// visibility when the HFI deserialization is triggered by a sibling submodule.
//
// The scenario:
//   Module A contains A.h, which #includes non-modular Textual.h.
//   Module B contains B1.h and B2.h.
//   B1.h imports A.h, then imports B2.h (triggering B2's submodule processing),
//   then imports Textual.h.
//   B2.h imports A.h, then imports Textual.h (triggering HFI deserialization).
//
// The included header's visibility should be determined by program text, not
// related to implementation details such as HFI deserlization.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- include/Textual.h
struct my_struct {
  int field;
};

//--- include/module.modulemap
module A {
  header "A.h"
  export *
}

module B {
  module B1 { header "B1.h" export * }
  module B2 { header "B2.h" export * }
}

//--- include/A.h
#include "Textual.h"

//--- include/B1.h
#import "A.h"
#import "B2.h"
#import "Textual.h"
static struct my_struct x;

//--- include/B2.h
#import "A.h"
#import "Textual.h"
static struct my_struct y;

//--- tu.c
#import "B1.h"

// Build with implicit modules and local submodule visibility.
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -fimplicit-module-maps \
// RUN:   -fmodules-local-submodule-visibility -I %t/include -fsyntax-only %t/tu.c -verify

// expected-no-diagnostics
