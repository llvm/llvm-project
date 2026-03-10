// Test that #import re-enters textual headers included at multiple layers of
// a cross-module dependency chain when none of the including modules are
// visible.
//
// Setup:
//   Module A: A1 textually includes T1.h (no export *)
//   Module B: B1 imports A1 cross-module and textually includes T2.h (no export *)
//   Module C: C1 imports B1 cross-module (no export *)
//
// The TU imports C1, then #import T1.h and T2.h. Since no module in the chain
// has export *, neither A1 nor B1 is visible, so both T1.h and T2.h must be
// re-entered.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- T1.h
#define MACRO_T1 10

//--- T2.h
#define MACRO_T2 20

// =========================================================================
// Module A: A1 textually includes T1.h. NO export *.
// =========================================================================
//--- A/A1.h
#include "T1.h"
//--- A/module.modulemap
module A {
  module A1 { header "A1.h" }
}

// =========================================================================
// Module B: B1 imports A1 (cross-module) and textually includes T2.h.
// NO export *.
// =========================================================================
//--- B/B1.h
#include "A/A1.h"
#include "T2.h"
//--- B/module.modulemap
module B {
  module B1 { header "B1.h" }
}

// =========================================================================
// Module C: C1 imports B1 (cross-module). NO export *.
// =========================================================================
//--- C/C1.h
#include "B/B1.h"
//--- C/module.modulemap
module C {
  module C1 { header "C1.h" }
}

// =========================================================================
// Test: Import C1 → B1 → A1. No export * anywhere.
// Neither A1 nor B1 is visible, so both T1.h and T2.h must be re-entered.
// =========================================================================
//--- test.c
#import "C/C1.h"
#import "T1.h"
#import "T2.h"
static int x = MACRO_T1 + MACRO_T2;

// =========================================================================
// Build PCMs.
// =========================================================================
// RUN: %clang_cc1 -fmodules -I %t -emit-module %t/A/module.modulemap -fmodule-name=A -o %t/A.pcm
// RUN: %clang_cc1 -fmodules -I %t -emit-module %t/B/module.modulemap -fmodule-name=B -fmodule-file=%t/A.pcm -o %t/B.pcm
// RUN: %clang_cc1 -fmodules -I %t -emit-module %t/C/module.modulemap -fmodule-name=C -fmodule-file=%t/B.pcm -o %t/C.pcm

// =========================================================================
// Run test.
// =========================================================================
// RUN: %clang_cc1 -fmodules -I %t -fsyntax-only %t/test.c -fmodule-file=%t/C.pcm
