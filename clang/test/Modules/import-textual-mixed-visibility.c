// Test mixed visibility: one textual header is visible (via export *) and
// another is not. The visible one must be suppressed on #import, and the
// non-visible one must be re-entered.
//
// Setup:
//   Module A: A1 textually includes T1.h (export *)
//   Module B: B1 includes B2 (NO export *), B2 textually includes T2.h
//   Module C: C1 imports A1 and B1 cross-module (export *)
//
// The TU imports C1. Because C1 has export *:
//   - A1 is visible (C1 re-exports A1, and A1 has export *)
//   - B1 is visible (C1 re-exports B1)
//   - B2 is NOT visible (B1 has no export *, so B2 is not re-exported)
//
// Result:
//   T1.h included by A1 (visible): suppress
//   T2.h included by B2 (NOT visible): re-enter

// RUN: rm -rf %t
// RUN: split-file %s %t

// T1.h has a typedef — if incorrectly re-entered, duplicate typedef errors.
//--- T1.h
typedef int t1_type;
#define MACRO_T1 10

// T2.h has only a macro — if incorrectly suppressed, MACRO_T2 is undefined.
//--- T2.h
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
// Module B: B1 includes B2, B2 textually includes T2.h. B1 has NO export *.
// =========================================================================
//--- B/B1.h
#include "B2.h"
//--- B/B2.h
#include "T2.h"
//--- B/module.modulemap
module B {
  module B1 { header "B1.h" }
  module B2 { header "B2.h" }
}

// =========================================================================
// Module C: C1 imports A1 and B1 cross-module. Has export *.
// =========================================================================
//--- C/C1.h
#include "A/A1.h"
#include "B/B1.h"
//--- C/module.modulemap
module C {
  module C1 {
    header "C1.h"
    export *
  }
}

// =========================================================================
// Test: Import C1 (export *) → A1 (export *) visible, B1 visible but B2 not.
// T1.h included by A1 (visible) → suppress (typedef would error if re-entered).
// T2.h included by B2 (NOT visible) → re-enter (macro must be defined).
// =========================================================================
//--- test.c
#import "C/C1.h"
#import "T1.h"
#import "T2.h"
t1_type x = MACRO_T1;
static int y = MACRO_T2;

// =========================================================================
// Build PCMs.
// =========================================================================
// RUN: %clang_cc1 -fmodules -I %t -emit-module %t/A/module.modulemap -fmodule-name=A -o %t/A.pcm
// RUN: %clang_cc1 -fmodules -I %t -emit-module %t/B/module.modulemap -fmodule-name=B -o %t/B.pcm
// RUN: %clang_cc1 -fmodules -I %t -emit-module %t/C/module.modulemap -fmodule-name=C -fmodule-file=%t/A.pcm -fmodule-file=%t/B.pcm -o %t/C.pcm

// =========================================================================
// Run test.
// =========================================================================
// RUN: %clang_cc1 -fmodules -I %t -fsyntax-only %t/test.c -fmodule-file=%t/C.pcm
