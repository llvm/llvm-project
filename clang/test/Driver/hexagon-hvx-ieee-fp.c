// -----------------------------------------------------------------------------
// Tests for the hvx ieee fp feature and errors.
// -----------------------------------------------------------------------------

// RUN: %clang -c %s -### --target=hexagon-unknown-elf -mv68 -mhvx -mhvx-ieee-fp \
// RUN:   2>&1 | FileCheck -check-prefix=CHECK-IEEEFP %s
// RUN: %clang -c %s -### --target=hexagon-unknown-elf -mv66 -mhvx=v68 -mhvx-ieee-fp \
// RUN:   2>&1 | FileCheck -check-prefix=CHECK-IEEEFP %s
// CHECK-IEEEFP: "-target-feature" "+hvx-ieee-fp"

// RUN: %clang -c %s -### --target=hexagon-unknown-elf -mv68 -mhvx -mhvx-ieee-fp \
// RUN:  -mno-hvx-ieee-fp 2>&1 | FileCheck -check-prefix=CHECK-NO-IEEEFP %s
// CHECK-NO-IEEEFP: "-target-feature" "-hvx-ieee-fp"

// IEEE-FP is valid only on hvxv68 and hvxv68+.
// RUN: not %clang -c %s -### --target=hexagon-unknown-elf -mv68 -mhvx=v66 \
// RUN:   -mhvx-ieee-fp 2>&1 | FileCheck -check-prefix=CHECK-ERROR1 %s
// RUN: not %clang -c %s -### --target=hexagon-unknown-elf -mv66 -mhvx -mhvx-ieee-fp \
// RUN:   2>&1 | FileCheck -check-prefix=CHECK-ERROR1 %s
// CHECK-ERROR1: error: -mhvx-ieee-fp is not supported on HVX v66

// IEEE-FP is valid only if HVX is enabled.
// RUN: not %clang -c %s -### --target=hexagon-unknown-elf -mv68 -mhvx-ieee-fp \
// RUN:   2>&1 | FileCheck -check-prefix=CHECK-ERROR2 %s
// CHECK-ERROR2: error: -mhvx-ieee-fp requires HVX, use -mhvx/-mhvx= to enable it
