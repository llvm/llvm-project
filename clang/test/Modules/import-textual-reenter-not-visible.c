// Test that #import can re-enter a textual header when the header was included
// by a module that is reachable but NOT visible. The key invariant:
//
//   If a textual header was included by module M, and M is NOT visible in the
//   current TU, then #import of that header MUST re-enter the file (not skip it).
//
// This is the complement of import-textual-skip-visible.c, which verifies
// the opposite: visible modules DO suppress re-entry. Here we verify that
// non-visible modules do NOT suppress re-entry, even when the PCM containing
// the header info has been loaded.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- Textual.h
#define MACRO_TEXTUAL 1

// =========================================================================
// Module E: E1 includes E2, E2 includes Textual.h. NO export *.
// =========================================================================
//--- E/E1.h
#include "E2.h"
//--- E/E2.h
#include "Textual.h"
//--- E/module.modulemap
module E {
  module E1 { header "E1.h" }
  module E2 { header "E2.h" }
}

// =========================================================================
// Module F: F1 imports E1 (cross-module via -fmodule-file). NO export *.
// =========================================================================
//--- F/F1.h
#include "E/E1.h"
//--- F/module.modulemap
module F {
  module F1 { header "F1.h" }
}

// =========================================================================
// Module G: G1 imports E1 with export *, but E1 itself does NOT export E2.
// So importing G1 makes E1 visible (via G1's export *), but E2 is still not
// visible because E1 has no export *.
// =========================================================================
//--- G/G1.h
#include "E/E1.h"
//--- G/module.modulemap
module G {
  module G1 {
    header "G1.h"
    export *
  }
}

// =========================================================================
// Module H: H1 imports E1, H2 is empty. Import H2 only — E2 is not visible.
// =========================================================================
//--- H/H1.h
#include "E/E1.h"
//--- H/H2.h
// empty
//--- H/module.modulemap
module H {
  module H1 { header "H1.h" }
  module H2 { header "H2.h" }
}

// =========================================================================
// Module J: another independent module that also includes Textual.h.
// =========================================================================
//--- J/J1.h
#include "Textual.h"
//--- J/module.modulemap
module J {
  module J1 { header "J1.h" }
}

// =========================================================================
// Test 1: Cross-module, no export.
// Import F1, which imports E1 (no export *). E2 included Textual.h but is
// NOT visible. #import Textual.h must re-enter the file.
// =========================================================================
//--- test_cross_no_export.c
#import "F/F1.h"
#import "Textual.h"
static int x = MACRO_TEXTUAL;

// =========================================================================
// Test 2: Cross-module, partial export chain.
// Import G1 (export *) → E1 (no export *). E1 is visible but E2 is not.
// Textual.h was included by E2 — must be re-enterable.
// =========================================================================
//--- test_partial_export.c
#import "G/G1.h"
#import "Textual.h"
static int x = MACRO_TEXTUAL;

// =========================================================================
// Test 3: Import a sibling; the other sibling's transitive deps included
// Textual.h. Import H2 only — H1 (and its dep E1, E2) are not visible.
// Textual.h must be re-enterable.
// =========================================================================
//--- test_sibling_invisible.c
#import "H/H2.h"
#import "Textual.h"
static int x = MACRO_TEXTUAL;

// =========================================================================
// Test 4: Multiple PCMs loaded, none with Textual.h visible.
// Load both E.pcm and J.pcm. Import neither E2 nor J1.
// Textual.h was included by both E2 and J1, but neither is visible.
// Must be re-enterable.
// =========================================================================
//--- test_multiple_pcms.c
// Don't import anything — just load PCMs via -fmodule-file.
#import "Textual.h"
static int x = MACRO_TEXTUAL;

// =========================================================================
// Test 5: Cross-module transitive, no export at any level.
// Import F1 → E1 → E2 → Textual.h. Neither F1 nor E1 has export *.
// E2 is not visible. Textual.h must be re-enterable.
// =========================================================================
//--- test_deep_no_export.c
#import "F/F1.h"
#import "Textual.h"
static int x = MACRO_TEXTUAL;

// =========================================================================
// Build PCMs.
// =========================================================================
// RUN: %clang_cc1 -fmodules -I %t -emit-module %t/E/module.modulemap -fmodule-name=E -o %t/E.pcm
// RUN: %clang_cc1 -fmodules -I %t -emit-module %t/F/module.modulemap -fmodule-name=F -fmodule-file=%t/E.pcm -o %t/F.pcm
// RUN: %clang_cc1 -fmodules -I %t -emit-module %t/G/module.modulemap -fmodule-name=G -fmodule-file=%t/E.pcm -o %t/G.pcm
// RUN: %clang_cc1 -fmodules -I %t -emit-module %t/H/module.modulemap -fmodule-name=H -fmodule-file=%t/E.pcm -o %t/H.pcm
// RUN: %clang_cc1 -fmodules -I %t -emit-module %t/J/module.modulemap -fmodule-name=J -o %t/J.pcm

// =========================================================================
// Run tests.
// =========================================================================

// Test 1: cross-module, no export — Textual.h must be re-entered.
// RUN: %clang_cc1 -fmodules -I %t -fsyntax-only %t/test_cross_no_export.c -fmodule-file=%t/F.pcm

// Test 2: partial export chain — Textual.h must be re-entered.
// RUN: %clang_cc1 -fmodules -I %t -fsyntax-only %t/test_partial_export.c -fmodule-file=%t/G.pcm

// Test 3: sibling invisible — Textual.h must be re-entered.
// RUN: %clang_cc1 -fmodules -I %t -fsyntax-only %t/test_sibling_invisible.c -fmodule-file=%t/H.pcm

// Test 4: multiple PCMs, none visible — Textual.h must be re-entered.
// RUN: %clang_cc1 -fmodules -I %t -fsyntax-only %t/test_multiple_pcms.c -fmodule-file=%t/E.pcm -fmodule-file=%t/J.pcm

// Test 5: deep transitive, no export — Textual.h must be re-entered.
// RUN: %clang_cc1 -fmodules -I %t -fsyntax-only %t/test_deep_no_export.c -fmodule-file=%t/F.pcm
