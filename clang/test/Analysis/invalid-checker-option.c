// RUN: not %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config RetainOneTwoThree:CheckOSObject=false \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-NON-EXISTENT-CHECKER

// Note that non-existent packages and checkers were always reported.

// RUN: not %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config-compatibility-mode=true \
// RUN:   -analyzer-config RetainOneTwoThree:CheckOSObject=false \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-NON-EXISTENT-CHECKER

// CHECK-NON-EXISTENT-CHECKER: (frontend): no analyzer checkers or packages
// CHECK-NON-EXISTENT-CHECKER-SAME: are associated with 'RetainOneTwoThree'


// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ConfigDumper \
// RUN:   -analyzer-checker=debug.AnalysisOrder \
// RUN:   -analyzer-config-compatibility-mode=true \
// RUN:   -analyzer-config debug.AnalysisOrder:*=yesplease \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-CORRECTED-BOOL-VALUE

// CHECK-CORRECTED-BOOL-VALUE: debug.AnalysisOrder:* = false
//
// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ConfigDumper \
// RUN:   -analyzer-checker=optin.performance.Padding \
// RUN:   -analyzer-config-compatibility-mode=true \
// RUN:   -analyzer-config optin.performance.Padding:AllowedPad=surpriseme \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-CORRECTED-INT-VALUE

// CHECK-CORRECTED-INT-VALUE: optin.performance.Padding:AllowedPad = 24


// Every other error should be avoidable in compatiblity mode.


// RUN: not %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config debug.AnalysisOrder:Everything=false \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-NON-EXISTENT-CHECKER-OPTION

// CHECK-NON-EXISTENT-CHECKER-OPTION: (frontend): checker 'debug.AnalysisOrder'
// CHECK-NON-EXISTENT-CHECKER-OPTION-SAME: has no option called 'Everything'

// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config-compatibility-mode=true \
// RUN:   -analyzer-config debug.AnalysisOrder:Everything=false


// RUN: not %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config debug.AnalysisOrder:*=nothankyou \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-INVALID-BOOL-VALUE

// CHECK-INVALID-BOOL-VALUE: (frontend): invalid input for checker option
// CHECK-INVALID-BOOL-VALUE-SAME: 'debug.AnalysisOrder:*', that expects a
// CHECK-INVALID-BOOL-VALUE-SAME: boolean value

// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config-compatibility-mode=true \
// RUN:   -analyzer-config debug.AnalysisOrder:*=nothankyou


// RUN: not %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config optin.performance.Padding:AllowedPad=surpriseme \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-INVALID-INT-VALUE

// CHECK-INVALID-INT-VALUE: (frontend): invalid input for checker option
// CHECK-INVALID-INT-VALUE-SAME: 'optin.performance.Padding:AllowedPad', that
// CHECK-INVALID-INT-VALUE-SAME: expects an integer value

// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config-compatibility-mode=true \
// RUN:   -analyzer-config optin.performance.Padding:AllowedPad=surpriseme


// RUN: not %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config nullability:NoDiagnoseCallsToSystemHeaders=sure \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-PACKAGE-VALUE

// CHECK-PACKAGE-VALUE: (frontend): invalid input for checker option
// CHECK-PACKAGE-VALUE-SAME: 'nullability:NoDiagnoseCallsToSystemHeaders', that
// CHECK-PACKAGE-VALUE-SAME: expects a boolean value

// expected-no-diagnostics

int main(void) {}
