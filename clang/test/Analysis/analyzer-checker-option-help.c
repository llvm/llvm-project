// RUN: %clang_cc1 -analyzer-checker-option-help \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-STABLE

// RUN: %clang_cc1 -analyzer-checker-option-help-alpha \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-ALPHA

// RUN: %clang_cc1 -analyzer-checker-option-help-developer \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-DEVELOPER

// RUN: %clang_cc1 -analyzer-checker-option-help-developer \
// RUN:   -analyzer-checker-option-help-alpha \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-DEVELOPER-ALPHA

// RUN: %clang_cc1 -analyzer-checker-option-help \
// RUN:   -analyzer-checker-option-help-alpha \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-STABLE-ALPHA

// RUN: %clang_cc1 -analyzer-checker-option-help \
// RUN:   -analyzer-checker-option-help-developer \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-STABLE-DEVELOPER

// RUN: %clang_cc1 -analyzer-checker-option-help \
// RUN:   -analyzer-checker-option-help-alpha \
// RUN:   -analyzer-checker-option-help-developer \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-STABLE-ALPHA-DEVELOPER

// CHECK-STABLE: OVERVIEW: Clang Static Analyzer Checker and Package Option List
//
// CHECK-STABLE: USAGE: -analyzer-config <OPTION1=VALUE,OPTION2=VALUE,...>
//
// CHECK-STABLE:        -analyzer-config OPTION1=VALUE, -analyzer-config
// CHECK-STABLE-SAME:   OPTION2=VALUE, ...
//
// CHECK-STABLE: OPTIONS:
//
// CHECK-STABLE:   cplusplus.Move:WarnOn
// CHECK-STABLE-SAME:          (string) In non-aggressive mode, only warn
// CHECK-STABLLE:              on use-after-move of local variables (or
// CHECK-STABLLE:              local rvalue references) and of STL objects.
// CHECK-STABLLE:              The former is possible because local variables
// CHECK-STABLLE:              (or local rvalue references) are not tempting
// CHECK-STABLLE:              their user to re-use the storage. The latter
// CHECK-STABLLE:              is possible because STL objects are known
// CHECK-STABLLE:              to end up in a valid but unspecified state
// CHECK-STABLLE:              after the move and their state-reset methods
// CHECK-STABLLE:              are also known, which allows us to predict
// CHECK-STABLLE:              precisely when use-after-move is invalid.
// CHECK-STABLLE:              Some STL objects are known to conform to
// CHECK-STABLLE:              additional contracts after move, so they
// CHECK-STABLLE:              are not tracked. However, smart pointers
// CHECK-STABLLE:              specifically are tracked because we can
// CHECK-STABLLE:              perform extra checking over them. In aggressive
// CHECK-STABLLE:              mode, warn on any use-after-move because
// CHECK-STABLLE:              the user has intentionally asked us to completely
// CHECK-STABLLE:              eliminate use-after-move in his code. Values:
// CHECK-STABLLE:              "KnownsOnly", "KnownsAndLocals", "All".
// CHECK-STABLLE:              (default: KnownsAndLocals)

// CHECK-STABLE-NOT: debug.AnalysisOrder:*
// CHECK-DEVELOPER:  debug.AnalysisOrder:*
// CHECK-ALPHA-NOT:  debug.AnalysisOrder:*

// CHECK-STABLE-NOT:    optin.cplusplus.UninitializedObject:IgnoreGuardedFields
// CHECK-DEVELOPER-NOT: optin.cplusplus.UninitializedObject:IgnoreGuardedFields
// CHECK-ALPHA:         optin.cplusplus.UninitializedObject:IgnoreGuardedFields

// CHECK-STABLE:        optin.performance.Padding:AllowedPad
// CHECK-DEVELOPER-NOT: optin.performance.Padding:AllowedPad
// CHECK-ALPHA-NOT:     optin.performance.Padding:AllowedPad


// CHECK-STABLE-ALPHA-NOT: debug.AnalysisOrder:*
// CHECK-DEVELOPER-ALPHA:  debug.AnalysisOrder:*
// CHECK-STABLE-DEVELOPER: debug.AnalysisOrder:*

// CHECK-STABLE-ALPHA:         optin.cplusplus.UninitializedObject:IgnoreGuardedFields
// CHECK-DEVELOPER-ALPHA:      optin.cplusplus.UninitializedObject:IgnoreGuardedFields
// CHECK-STABLE-DEVELOPER-NOT: optin.cplusplus.UninitializedObject:IgnoreGuardedFields

// CHECK-STABLE-ALPHA:        optin.performance.Padding:AllowedPad
// CHECK-DEVELOPER-ALPHA-NOT: optin.performance.Padding:AllowedPad
// CHECK-STABLE-DEVELOPER:    optin.performance.Padding:AllowedPad


// CHECK-STABLE-ALPHA-DEVELOPER: debug.AnalysisOrder:*
// CHECK-STABLE-ALPHA-DEVELOPER: optin.cplusplus.UninitializedObject:IgnoreGuardedFields
// CHECK-STABLE-ALPHA-DEVELOPER: optin.performance.Padding:AllowedPad
