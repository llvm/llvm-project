// Test that #import does NOT re-enter textual headers included at multiple
// layers of a cross-module dependency chain when the including modules ARE
// visible (via transitive export *).
//
// Setup:
//   Module A: A1 textually includes T1.h (export *)
//   Module B: B1 imports A1 cross-module and textually includes T2.h (export *)
//   Module C: C1 imports B1 cross-module (export *)
//
// The TU imports C1. Because every module in the chain has export *, B1 and
// A1 are transitively visible. T1.h was included by A1 and T2.h was included
// by B1 — both visible. So #import of T1.h and T2.h must be suppressed.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- T1.h
typedef int t1_type;
#define MACRO_T1 10

//--- T2.h
typedef int t2_type;
#define MACRO_T2 20

// =========================================================================
// Module A: A1 textually includes T1.h. Has export *.
// =========================================================================
//--- A/A1.h
#include "T1.h"
//--- A/module.modulemap
module A {
  module A1 {
    header "A1.h"
    export *
  }
}

// =========================================================================
// Module B: B1 imports A1 (cross-module) and textually includes T2.h.
// Has export *.
// =========================================================================
//--- B/B1.h
#include "A/A1.h"
#include "T2.h"
//--- B/module.modulemap
module B {
  module B1 {
    header "B1.h"
    export *
  }
}

// =========================================================================
// Module C: C1 imports B1 (cross-module). Has export *.
// =========================================================================
//--- C/C1.h
#include "B/B1.h"
//--- C/module.modulemap
module C {
  module C1 {
    header "C1.h"
    export *
  }
}

// =========================================================================
// Test: Import C1 (export *) → B1 (export *) → A1 (export *).
// A1 and B1 are both visible. T1.h and T2.h were included by visible
// modules, so #import must be suppressed. The typedefs in T1.h and T2.h
// (no include guard) would cause duplicate definition errors if the files
// were incorrectly re-entered.
// =========================================================================
//--- test.c
#import "C/C1.h"
#import "T1.h"
#import "T2.h"
t1_type x = MACRO_T1;
t2_type y = MACRO_T2;

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
