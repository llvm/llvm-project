// Test that #import skips a non-modular header when the header is included
// by a submodule that is visible. The key invariant:
//
//   If a non-modular header was included by module M, and M is visible in the
//   current TU, then #import of that header skips importing the header. In
//   other words, the #import sematic (import only once) is satisfied.
//
// We cover two different cases, and we skip the header in the
// translation unit for both of them:
// 1. We import a submodule that includes the non-modular header.
// 2. We import a submodule that transitively imports a visible submodule
//    that includes the non-modular header.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- include/NonModular.h
typedef int non_modular_type_t;
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
  module B1 { header "B1.h" export * }
  module B2 { header "B2.h" }
}

//--- include/B/B1.h
#include "A/A1.h"

//--- include/B/B2.h

//--- test_direct_import.c
#import "A/A1.h"
#import "NonModular.h"
non_modular_type_t val = MACRO_NON_MODULAR;

//--- test_transitive_import.c
#import "B/B1.h"
#import "NonModular.h"
non_modular_type_t val = MACRO_NON_MODULAR;

// Build the pcms
// RUN: %clang_cc1 -fmodules -I %t/include -emit-module %t/include/A/module.modulemap -fmodule-name=A -o %t/A.pcm
// RUN: %clang_cc1 -fmodules -I %t/include -emit-module %t/include/B/module.modulemap -fmodule-name=B -o %t/B.pcm

// Test 1: directly importing a submodule that includes a non-modular header.
// RUN: %clang_cc1 -fmodules -I %t/include -fsyntax-only %t/test_direct_import.c -fmodule-file=%t/A.pcm

// Test 2: importing a submodule that transitively imports a visible submodule that includes a non-modular header.
// RUN: %clang_cc1 -fmodules -I %t/include -fsyntax-only %t/test_transitive_import.c -fmodule-file=%t/B.pcm
