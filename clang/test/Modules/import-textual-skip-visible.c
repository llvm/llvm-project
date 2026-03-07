// Test that #import does NOT re-enter a textual header when the header was
// included by a module that IS visible. The key invariant:
//
//   If a textual header was included by module M, and M is visible in the
//   current TU (directly or via transitive export), then #import of that
//   header MUST be skipped (the file is already included).
//
// This is the complement of import-textual-reenter-not-visible.c, which
// verifies the opposite: non-visible modules do NOT suppress re-entry.

// RUN: rm -rf %t
// RUN: split-file %s %t

// Textual.h has no include guard. If #import incorrectly re-enters the file,
// the duplicate typedef will produce a compilation error.
//--- Textual.h
typedef int textual_type_t;
#define MACRO_TEXTUAL 1

// =========================================================================
// Module E: E1 includes E2, E2 includes Textual.h. E1 has export *.
// =========================================================================
//--- E/E1.h
#include "E2.h"
//--- E/E2.h
#include "Textual.h"
//--- E/module.modulemap
module E {
  module E1 {
    header "E1.h"
    export *
  }
  module E2 { header "E2.h" }
}

// =========================================================================
// Module F: F1 imports E1 cross-module. F1 has export *.
// =========================================================================
//--- F/F1.h
#include "E/E1.h"
//--- F/module.modulemap
module F {
  module F1 {
    header "F1.h"
    export *
  }
}

// =========================================================================
// Module G: G1 imports F1 cross-module. G1 has export *.
// =========================================================================
//--- G/G1.h
#include "F/F1.h"
//--- G/module.modulemap
module G {
  module G1 {
    header "G1.h"
    export *
  }
}

// =========================================================================
// Module A: A1 independently includes Textual.h (for early deserialization).
// =========================================================================
//--- A/A1.h
#include "Textual.h"
//--- A/module.modulemap
module A {
  module A1 { header "A1.h" }
}

// =========================================================================
// Test 1: Sibling re-export within a single module.
// Import E1 (export *) → E2 becomes visible → Textual.h's macro should
// be available as a module macro without an explicit #import.
// =========================================================================
//--- test_sibling_reexport.c
#import "E/E1.h"
#ifdef MACRO_TEXTUAL
textual_type_t val = MACRO_TEXTUAL;
#else
#error "MACRO_TEXTUAL should be visible via E1's re-export of E2"
#endif

// =========================================================================
// Test 2: Cross-module transitive re-export (depth 2).
// Import F1 (export *) → E1 (export *) → E2 visible. Macro should be
// available without explicit #import.
// =========================================================================
//--- test_cross_depth2.c
#import "F/F1.h"
#ifdef MACRO_TEXTUAL
textual_type_t val = MACRO_TEXTUAL;
#else
#error "MACRO_TEXTUAL should be visible via F1 -> E1 -> E2 export chain"
#endif

// =========================================================================
// Test 3: Cross-module transitive re-export (depth 3).
// Import G1 (export *) → F1 (export *) → E1 (export *) → E2 visible.
// =========================================================================
//--- test_cross_depth3.c
#import "G/G1.h"
#ifdef MACRO_TEXTUAL
textual_type_t val = MACRO_TEXTUAL;
#else
#error "MACRO_TEXTUAL should be visible via G1 -> F1 -> E1 -> E2 export chain"
#endif

// =========================================================================
// Test 4: Explicit #import of already-visible textual header is suppressed.
// Import E1 (export *) → E2 visible → Textual.h already included.
// A subsequent #import Textual.h should be a no-op. The typedef in
// Textual.h (no include guard) would cause a duplicate definition error
// if the file were incorrectly re-entered.
// =========================================================================
//--- test_import_suppressed.c
#import "E/E1.h"
#import "Textual.h"
textual_type_t val = MACRO_TEXTUAL;

// =========================================================================
// Test 5: Cross-module #import suppression (depth 2).
// Import F1 (export * chain) → E2 visible. Explicit #import of Textual.h
// must be suppressed.
// =========================================================================
//--- test_import_suppressed_depth2.c
#import "F/F1.h"
#import "Textual.h"
textual_type_t val = MACRO_TEXTUAL;

// =========================================================================
// Test 6: Cross-module #import suppression (depth 3).
// Import G1 (export * chain) → E2 visible. Explicit #import of Textual.h
// must be suppressed.
// =========================================================================
//--- test_import_suppressed_depth3.c
#import "G/G1.h"
#import "Textual.h"
textual_type_t val = MACRO_TEXTUAL;

// =========================================================================
// Test 7: Early deserialization — import a different module that also
// includes Textual.h BEFORE importing E1. This forces Textual.h's header
// info to be deserialized (via markIncludedInModule) at a time when E2 is
// NOT yet visible. Then import E1 (export *) → E2 becomes visible.
// The subsequent #import Textual.h must still be suppressed.
// =========================================================================
//--- test_early_deser.c
#import "A/A1.h"
#import "E/E1.h"
#import "Textual.h"
textual_type_t val = MACRO_TEXTUAL;

// =========================================================================
// Build PCMs.
// =========================================================================
// RUN: %clang_cc1 -fmodules -I %t -emit-module %t/E/module.modulemap -fmodule-name=E -o %t/E.pcm
// RUN: %clang_cc1 -fmodules -I %t -emit-module %t/F/module.modulemap -fmodule-name=F -fmodule-file=%t/E.pcm -o %t/F.pcm
// RUN: %clang_cc1 -fmodules -I %t -emit-module %t/G/module.modulemap -fmodule-name=G -fmodule-file=%t/F.pcm -o %t/G.pcm
// RUN: %clang_cc1 -fmodules -I %t -emit-module %t/A/module.modulemap -fmodule-name=A -o %t/A.pcm

// =========================================================================
// Run tests.
// =========================================================================

// Test 1: sibling re-export — macro visible without #import.
// RUN: %clang_cc1 -fmodules -I %t -fsyntax-only %t/test_sibling_reexport.c -fmodule-file=%t/E.pcm

// Test 2: depth-2 cross-module re-export — macro visible.
// RUN: %clang_cc1 -fmodules -I %t -fsyntax-only %t/test_cross_depth2.c -fmodule-file=%t/F.pcm

// Test 3: depth-3 cross-module re-export — macro visible.
// RUN: %clang_cc1 -fmodules -I %t -fsyntax-only %t/test_cross_depth3.c -fmodule-file=%t/G.pcm

// Test 4: explicit #import suppressed (single module).
// RUN: %clang_cc1 -fmodules -I %t -fsyntax-only %t/test_import_suppressed.c -fmodule-file=%t/E.pcm

// Test 5: explicit #import suppressed (depth 2).
// RUN: %clang_cc1 -fmodules -I %t -fsyntax-only %t/test_import_suppressed_depth2.c -fmodule-file=%t/F.pcm

// Test 6: explicit #import suppressed (depth 3).
// RUN: %clang_cc1 -fmodules -I %t -fsyntax-only %t/test_import_suppressed_depth3.c -fmodule-file=%t/G.pcm

// Test 7: early deserialization — #import suppressed despite prior deser.
// RUN: %clang_cc1 -fmodules -I %t -fsyntax-only %t/test_early_deser.c -fmodule-file=%t/A.pcm -fmodule-file=%t/E.pcm
