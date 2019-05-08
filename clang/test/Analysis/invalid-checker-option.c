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

// expected-no-diagnostics

int main() {}
