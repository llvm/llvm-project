// Test that #import correctly skips a non-modular header under local submodule
// visibility.
//
// The scenario:
//   Module A contains A.h, which #includes non-modular NonModular.h.
//   Module B contains B1.h and B2.h, both of which #import A.h and then
//   #import NonModular.h.
//
// The complication happens when we build module B. B2 should obey the #import
// semantic regarding #import "NonModular.h": since A is visible, B2 should
// correctly identify that "NonModular.h" is already included through A, and
// skip the import in B2.h.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- include/NonModular.h
struct my_struct {
  int field;
};

//--- include/A/module.modulemap
module A {
  header "A.h"
  export *
}

//--- include/A/A.h
#include "NonModular.h"

//--- include/B/module.modulemap
module B {
  module B1 { header "B1.h" export * }
  module B2 { header "B2.h" export * }
}

//--- include/B/B1.h
#import "A/A.h"
#import "NonModular.h"
static struct my_struct x;

//--- include/B/B2.h
#import "A/A.h"
#import "NonModular.h"
static struct my_struct y;

//--- test.c
#import "B/B1.h"
#import "B/B2.h"

// Build module A
// RUN: %clang_cc1 -fmodules -I %t/include -emit-module %t/include/A/module.modulemap -fmodule-name=A -o %t/A.pcm

// Build module B with local submodule visibility, which gives each submodule
// its own CurSubmoduleState.
// RUN: %clang_cc1 -fmodules -fmodules-local-submodule-visibility -I %t/include -emit-module %t/include/B/module.modulemap -fmodule-name=B -fmodule-file=%t/A.pcm -o %t/B.pcm

// Use module B.
// RUN: %clang_cc1 -fmodules -fmodules-local-submodule-visibility -I %t/include -fsyntax-only %t/test.c -fmodule-file=%t/B.pcm -verify

// expected-no-diagnostics
