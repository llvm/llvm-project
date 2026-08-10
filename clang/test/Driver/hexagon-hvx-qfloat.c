// -----------------------------------------------------------------------------
// Tests for the hvx qfloat feature and errors.
// -----------------------------------------------------------------------------

// RUN: %clang -c %s -### --target=hexagon-unknown-elf -mv68 -mhvx -mhvx-qfloat \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-QFLOAT %s
// RUN: %clang -c %s -### --target=hexagon-unknown-elf -mv66 -mhvx=v68 -mhvx-qfloat \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-QFLOAT %s
// CHECK-QFLOAT: "-target-feature" "+hvx-qfloat"

// RUN: %clang -c %s -### --target=hexagon-unknown-elf -mv68 -mhvx -mhvx-qfloat \
// RUN:  -mno-hvx-qfloat 2>&1 | FileCheck -check-prefix=CHECK-NO-QFLOAT %s
// CHECK-NO-QFLOAT: "-target-feature" "-hvx-qfloat"

// QFloat is valid only on hvxv68+.
// RUN: not %clang -c %s -### --target=hexagon-unknown-elf -mv68 -mhvx=v66 \
// RUN: -mhvx-qfloat 2>&1 | FileCheck -check-prefix=CHECK-ERROR1 %s
// RUN: not %clang -c %s -### --target=hexagon-unknown-elf -mv66 -mhvx -mhvx-qfloat \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-ERROR1 %s
// CHECK-ERROR1: error: -mhvx-qfloat is not supported on HVX v66

// QFloat is valid only if HVX is enabled.
// RUN: not %clang -c %s -### --target=hexagon-unknown-elf -mv68 -mhvx-qfloat \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-ERROR2 %s
// CHECK-ERROR2: error: -mhvx-qfloat requires HVX, use -mhvx/-mhvx= to enable it
