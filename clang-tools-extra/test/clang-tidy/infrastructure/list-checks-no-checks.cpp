// RUN: not clang-tidy --list-checks -checks='-*' -- 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-ERROR
// CHECK-ERROR: No checks enabled.

// RUN: clang-tidy --list-checks -checks='-*' --allow-no-checks -- 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-OK
// CHECK-OK: Enabled checks:
