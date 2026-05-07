// Test that #import can re-enter a non-modular header when the header was included
// by a module that is reachable but NOT visible. The key invariant:
//
//   If a non-modular header was included by module M, and M is NOT visible in the
//   current TU, then #import of that header re-enters the header (not skip it).
//
// We cover three different cases, and we re-enter the header in the
// translation unit for all of them:
// 1. We load a pcm file of a module that includes the non-modular header. The
//    module is not imported, hence is invisible.
// 2. We import a sibling submodule of a submodule that includes the non-modular
//    header.
// 3. We import a submodule whose sibling transitively imports a module that
//    includes the non-modular header.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- include/NonModular.h
#define MACRO_NON_MODULAR 1

// module A's submodule A1 includes the non-modular header NonModular.h
//--- include/A/module.modulemap
module A {
  module A1 { header "A1.h" }
  module A2 { header "A2.h" }
}

//--- include/A/A1.h
#include "NonModular.h"

//--- include/A/A2.h

// module B transitively imports module A through header B1.h.
//--- include/B/module.modulemap
module B {
  module B1 { header "B1.h" }
  module B2 { header "B2.h" }
}

//--- include/B/B1.h
#include "A/A1.h"

//--- include/B/B2.h

//--- test_pcm_loaded.c
#import "NonModular.h"
static int x = MACRO_NON_MODULAR;

//--- test_invisible_sibling.c
#import "A/A2.h"
#import "NonModular.h"
static int x = MACRO_NON_MODULAR;

//--- test_invisible_transitive.c
#import "B/B2.h"
#import "NonModular.h"
static int x = MACRO_NON_MODULAR;

// Build the pcms
// RUN: %clang_cc1 -fmodules -I %t/include -emit-module %t/include/A/module.modulemap -fmodule-name=A -o %t/A.pcm
// RUN: %clang_cc1 -fmodules -I %t/include -emit-module %t/include/B/module.modulemap -fmodule-name=B -fmodule-file=%t/A.pcm -o %t/B.pcm

// Test case 1: loading the pcm but not importing it in the TU.
// RUN: %clang_cc1 -fmodules -I %t/include -fsyntax-only %t/test_pcm_loaded.c -fmodule-file=%t/A.pcm

// Test case 2: invisible sibling.
// RUN: %clang_cc1 -fmodules -I %t/include -fsyntax-only %t/test_invisible_sibling.c -fmodule-file=%t/A.pcm

// Test case 3: invisible transitive imported module.
// RUN: %clang_cc1 -fmodules -I %t/include -fsyntax-only %t/test_invisible_transitive.c -fmodule-file=%t/B.pcm
