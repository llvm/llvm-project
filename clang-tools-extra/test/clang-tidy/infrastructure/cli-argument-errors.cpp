
// RUN: not clang-tidy -- 2>&1 | FileCheck %s -check-prefix=CHECK-NO-INPUT
// CHECK-NO-INPUT: Error: no input files specified.

// RUN: not clang-tidy -checks='-*' %s -- 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-NO-CHECKS
// CHECK-NO-CHECKS: Error: no checks enabled.

// OK with --allow-no-checks
// RUN: clang-tidy -checks='-*' --allow-no-checks %s -- 2>&1 | count 0

// RUN: not clang-tidy --line-filter='not json' %s -- 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-LINE-FILTER
// CHECK-LINE-FILTER: Invalid LineFilter:
