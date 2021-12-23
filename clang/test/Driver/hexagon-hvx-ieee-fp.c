// -----------------------------------------------------------------------------
// Tests for the hvx ieee fp feature and errors.
// -----------------------------------------------------------------------------

// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv68 -mhvx -mhvx-ieee-fp \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-IEEEFP %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv66 -mhvx=v68 -mhvx-ieee-fp \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-IEEEFP %s
// CHECK-IEEEFP: "-target-feature" "+hvx-ieee-fp"

// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv68 -mhvx -mhvx-ieee-fp \
// RUN:  -mno-hvx-ieee-fp 2>&1 | FileCheck -check-prefix=CHECK-NO-IEEEFP %s
// CHECK-NO-IEEEFP: "-target-feature" "-hvx-ieee-fp"

